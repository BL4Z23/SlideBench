import json
import os
import random
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                  # falls back to the pure-Python search
    HAVE_NUMBA = False

ROOT =Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slidebench.engine import decode, move, valid_moves, won

# ---------------------------------------------------------------- settings
WORKERS = max(1, (os.cpu_count() or 2) - 1)
BATCH_SIZE = WORKERS * 4        # more tasks than workers keeps every core busy
ATTEMPTS = 10 ** 9              # safety cap only; a run ends when every bucket is full
PER_BUCKET = 200_000            # accepted boards per bucket (5 buckets -> 1M per run)
N_BUCKETS = 5
LO, HI = 5, 45                  # solution length range, split evenly into the buckets
CLIMB_ITERATIONS = 4000         # only searches that miss their target depth use them all
SAMPLES_PER_DEPTH = 50          # boards kept per depth, per harvested search
DEEP_FRACTION = 1.0             # share of searches aimed at the deepest open bucket (1.0 = hard boards only)
SEED_PROBABILITY = 0.9          # share of searches that start from a stored deep board, not a random one
PLATEAU_ITERATIONS = 500        # mutation steps for a search that starts from a stored deep board
SEED_POOL = 500                 # deep boards loaded as starting points
SEED_REFRESH_EVERY = 25         # batches between reloading that pool
TASK_CAP = 5000                 # a search stops once it has collected this many boards
VERIFY_FRACTION = 0.02          # share of new boards re-checked with the engine's BFS
MAX_STATES = 30000
REPORT_EVERY = 2000
COMMIT_EVERY = 2000
TOTAL_RUNS = 1
SEED = None

N_LENGTHS = HI - LO + 1         # 41 distinct solution lengths
LENGTH = {1: 2, 2: 2, 3: 3, 4: 2, 5: 3}
OUTPUT_DIR = ROOT / "data" / "generated"


def bucket_of(moves):
    """Map a solution length (LO..HI) to a bucket index (0..N_BUCKETS-1)."""
    return (moves - LO) * N_BUCKETS // N_LENGTHS


# (first_length, last_length) covered by each bucket
BUCKET_RANGES = [
    (
        min(m for m in range(LO, HI + 1) if bucket_of(m) == b),
        max(m for m in range(LO, HI + 1) if bucket_of(m) == b),
    )
    for b in range(N_BUCKETS)
]


# ------------------------------------------------------------ board search
RED_GOAL = 1 * 36 + 2 * 6 + 4
HORIZONTAL = (1, 2, 3)
MASKS = {}
MOVES = {}


def mask(code):
    m = MASKS.get(code)
    if m is None:
        v, pos = divmod(code, 36)
        if v in HORIZONTAL:
            m = ((1 << LENGTH[v]) - 1) << pos
        else:
            m = sum(1 << (pos + 6 * i) for i in range(LENGTH[v]))
        MASKS[code] = m
    return m


def move_table(code):
    """
    Precomputed slide options for one piece placement, built once per process.

    Returns (backward, forward, own_mask). Each direction is a tuple of
    (new_code, path_mask, new_mask) ordered by distance. path_mask is every
    cell swept to get there, so a slide is legal iff path_mask & other_pieces == 0.
    """
    t = MOVES.get(code)
    if t is None:
        v, pos = divmod(code, 36)
        n = LENGTH[v]
        r, c = divmod(pos, 6)
        if v in HORIZONTAL:
            back, forward, step = c, 5 - (c + n - 1), 1
        else:
            back, forward, step = r, 5 - (r + n - 1), 6

        directions = []
        for sign, limit in ((-1, back), (1, forward)):
            options = []
            path = 0
            for k in range(1, limit + 1):
                nx = code + sign * k * step
                path |= mask(nx)
                options.append((nx, path, mask(nx)))
            directions.append(tuple(options))

        t = MOVES[code] = (directions[0], directions[1], mask(code))
    return t


def expand(state, occ):
    """All states one slide away, each paired with its occupancy bitmask."""
    out = []
    for i, x in enumerate(state):
        back, forward, own = MOVES.get(x) or move_table(x)
        free = occ & ~own                      # everything except this piece
        rest = state[:i] + state[i + 1:]
        for options in (back, forward):
            for nx, path, nmask in options:
                if path & free:
                    break
                out.append((tuple(sorted(rest + (nx,))), free | nmask))
    return out


def evaluate_py(start):
    """Pure-Python reference search (also the fallback when numba is missing)."""
    occ0 = 0
    for x in start:
        occ0 |= mask(x)

    index = {start: 0}
    states = [start]
    occs = [occ0]
    adjacency = []

    i = 0
    while i < len(states):
        row = []
        for t, o in expand(states[i], occs[i]):
            j = index.get(t)
            if j is None:
                j = index[t] = len(states)
                states.append(t)
                occs.append(o)
                if j >= MAX_STATES:
                    return None
            row.append(j)
        adjacency.append(row)
        i += 1

    dist = [-1] * len(states)
    queue = deque()
    for i, s in enumerate(states):
        if RED_GOAL in s:
            dist[i] = 0
            queue.append(i)
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                queue.append(j)

    return np.array(states, dtype=np.uint8), np.array(dist, dtype=np.int32)


# ------------------------------------------------- numba-compiled search
# States are sorted uint8 piece codes. A state's hash-table slot is found by
# hashing those bytes, adjacency is stored CSR-style, and the distance BFS runs
# from every goal state, all without creating Python objects.
K_MAX = 5                                    # most slides a piece has in one direction
N_CODES = 6 * 36
TABLE_SIZE = 1 << (2 * MAX_STATES - 1).bit_length()


def build_move_arrays():
    own = np.zeros(N_CODES, dtype=np.uint64)
    cnt = np.zeros((N_CODES, 2), dtype=np.int64)
    nxt = np.zeros((N_CODES, 2, K_MAX), dtype=np.uint8)
    path = np.zeros((N_CODES, 2, K_MAX), dtype=np.uint64)
    nmask = np.zeros((N_CODES, 2, K_MAX), dtype=np.uint64)

    for v, n in LENGTH.items():
        for pos in range(36):
            r, c = divmod(pos, 6)
            if (c + n > 6) if v in HORIZONTAL else (r + n > 6):
                continue                     # piece would hang off the board
            code = v * 36 + pos
            back, forward, own_mask = move_table(code)
            own[code] = own_mask
            for d, options in enumerate((back, forward)):
                cnt[code, d] = len(options)
                for k, (nx, p, nm) in enumerate(options):
                    nxt[code, d, k] = nx
                    path[code, d, k] = p
                    nmask[code, d, k] = nm

    return own, cnt, nxt, path, nmask


if HAVE_NUMBA:

    MOVE_ARRAYS = build_move_arrays()

    @njit(cache=True)
    def _hash_row(row):
        h = np.uint64(14695981039346656037)
        for q in range(row.shape[0]):
            h = (h ^ np.uint64(row[q])) * np.uint64(1099511628211)
        return h

    @njit(cache=True)
    def _probe(table, states, tmp):
        n = tmp.shape[0]
        size_mask = np.int64(table.shape[0] - 1)
        slot = np.int64(_hash_row(tmp) & np.uint64(size_mask))
        while True:
            j = table[slot]
            if j < 0:
                return slot, -1
            same = True
            for q in range(n):
                if states[j, q] != tmp[q]:
                    same = False
                    break
            if same:
                return slot, j
            slot = (slot + 1) & size_mask

    @njit(cache=True)
    def _evaluate_nb(start, goal, max_states, table_size, own, cnt, nxt, path, nmask):
        n = start.shape[0]
        states = np.empty((max_states, n), dtype=np.uint8)
        occs = np.empty(max_states, dtype=np.uint64)
        table = np.full(table_size, -1, dtype=np.int32)
        adj_start = np.empty(max_states + 1, dtype=np.int32)
        cap = 1 << 18
        adj = np.empty(cap, dtype=np.int32)
        n_edges = 0
        tmp = np.empty(n, dtype=np.uint8)

        occ0 = np.uint64(0)
        for p in range(n):
            states[0, p] = start[p]
            tmp[p] = start[p]
            occ0 |= own[start[p]]
        occs[0] = occ0
        slot, found = _probe(table, states, tmp)
        table[slot] = 0
        count = 1

        i = 0
        while i < count:
            adj_start[i] = n_edges
            occ = occs[i]
            for p in range(n):
                x = states[i, p]
                free = occ & ~own[x]
                for d in range(2):
                    for k in range(cnt[x, d]):
                        if path[x, d, k] & free:
                            break
                        nx = nxt[x, d, k]

                        # new state = this one with piece p replaced, kept sorted
                        idx = 0
                        inserted = False
                        for q in range(n):
                            if q == p:
                                continue
                            v = states[i, q]
                            if (not inserted) and nx < v:
                                tmp[idx] = nx
                                idx += 1
                                inserted = True
                            tmp[idx] = v
                            idx += 1
                        if not inserted:
                            tmp[idx] = nx

                        slot, j = _probe(table, states, tmp)
                        if j < 0:
                            if count >= max_states:
                                return -1, states, np.empty(0, dtype=np.int32)
                            j = count
                            for q in range(n):
                                states[j, q] = tmp[q]
                            occs[j] = free | nmask[x, d, k]
                            table[slot] = j
                            count += 1

                        if n_edges == cap:
                            bigger = np.empty(cap * 2, dtype=np.int32)
                            bigger[:cap] = adj
                            adj = bigger
                            cap *= 2
                        adj[n_edges] = j
                        n_edges += 1
            i += 1
        adj_start[count] = n_edges

        dist = np.full(count, -1, dtype=np.int32)
        queue = np.empty(count, dtype=np.int32)
        head = 0
        tail = 0
        for i in range(count):
            for p in range(n):
                if states[i, p] == goal:
                    dist[i] = 0
                    queue[tail] = i
                    tail += 1
                    break
        while head < tail:
            i = queue[head]
            head += 1
            for e in range(adj_start[i], adj_start[i + 1]):
                j = adj[e]
                if dist[j] < 0:
                    dist[j] = dist[i] + 1
                    queue[tail] = j
                    tail += 1

        return count, states, dist

    def evaluate(start):
        """Returns (states array, distance array) or None if the space is too large."""
        arr = np.array(sorted(start), dtype=np.uint8)
        count, states, dist = _evaluate_nb(
            arr, RED_GOAL, MAX_STATES, TABLE_SIZE, *MOVE_ARRAYS
        )
        if count < 0:
            return None
        return states[:count], dist

else:
    evaluate = evaluate_py


def try_add(state, rng, v=None):
    occ = 0
    for x in state:
        occ |= mask(x)
    for _ in range(30):
        vv = v or int(rng.integers(2, 6))
        n = LENGTH[vv]
        if vv in HORIZONTAL:
            r, c = int(rng.integers(0, 6)), int(rng.integers(0, 6 - n + 1))
        else:
            r, c = int(rng.integers(0, 6 - n + 1)), int(rng.integers(0, 6))
        x = vv * 36 + r * 6 + c
        if not mask(x) & occ:
            return tuple(sorted(state + (x,)))
    return None


def random_start(rng):
    state = (RED_GOAL,)
    for _ in range(int(rng.integers(6, 12))):
        state = try_add(state, rng) or state
    return state


def mutate(state, rng):
    others = [x for x in state if x // 36 != 1]
    p = rng.random()
    if p < 0.3 and len(state) < 14:
        return try_add(state, rng)
    pieces = list(state)
    if p < 0.45 and len(others) > 4:
        pieces.remove(others[int(rng.integers(len(others)))])
        return tuple(pieces)
    x = others[int(rng.integers(len(others)))]
    pieces.remove(x)
    return try_add(tuple(pieces), rng, v=x // 36 if rng.random() < 0.6 else None)


def key_of_state(state):
    """36-character board key, identical to joining the flattened 6x6 grid."""
    cells = ["0"] * 36
    for x in state:
        v, pos = divmod(x, 36)
        step = 1 if v in HORIZONTAL else 6
        ch = str(v)
        for i in range(LENGTH[v]):
            cells[pos + i * step] = ch
    return "".join(cells)


def bfs_stripped(initial_board, max_moves=None):

    initial_board = tuple(
        (piece[0], tuple(piece[1]))
        for piece in sorted(initial_board, key=lambda p: p[1])
    )

    if won(initial_board):
        return {"solved": True, "moves": 0}

    visited = {initial_board}
    add = visited.add

    frontier = [initial_board]
    moves = 0

    while frontier:

        moves += 1

        if max_moves is not None and moves > max_moves:
            return {"solved": False, "moves": None}

        next_frontier = []
        append = next_frontier.append

        for board in frontier:

            for coordinates, distance in valid_moves(board):

                new_board = move(board, coordinates, distance)

                if new_board is None or new_board in visited:
                    continue

                if won(new_board):
                    return {"solved": True, "moves": moves}

                add(new_board)
                append(new_board)

        frontier = next_frontier

    return {"solved": False, "moves": None}


def depth_of(grid, max_moves):
    return bfs_stripped(decode(grid), max_moves=max_moves)["moves"]


def verify_key(key):
    """Independent engine BFS for one board (run only on a sample of new boards)."""
    grid = np.array(list(key), dtype=np.int8).reshape(6, 6)
    return depth_of(grid, HI + 1)


def board_to_state(key):
    """Inverse of key_of_state: rebuild the sorted piece codes from a 36-character board key."""
    cells = [int(ch) for ch in key]
    pieces = []
    for v, n in LENGTH.items():
        if v in HORIZONTAL:
            for r in range(6):
                c = 0
                while c < 6:
                    if cells[r * 6 + c] == v:
                        pieces.append(v * 36 + r * 6 + c)
                        c += n              # equal-length pieces of one type split a run uniquely
                    else:
                        c += 1
        else:
            for c in range(6):
                r = 0
                while r < 6:
                    if cells[r * 6 + c] == v:
                        pieces.append(v * 36 + r * 6 + c)
                        r += n
                    else:
                        r += 1
    return tuple(sorted(pieces))


def load_seed_starts(cur, tables, min_bucket, limit):
    """Random deep boards from this run and earlier runs, decoded into search start states."""
    keys = []
    for table in tables:
        keys.extend(
            row[0] for row in cur.execute(
                f"SELECT board FROM {table} WHERE accepted = 1 AND bucket >= ? "
                f"ORDER BY RANDOM() LIMIT ?",
                (min_bucket, limit),
            )
        )
    random.shuffle(keys)
    return [board_to_state(k) for k in keys[:limit]]


def harvest(found, result, rng, needed):
    """Sample boards from one fully explored state space, for buckets still open."""
    states, dist = result

    depths = [d for d in range(LO, HI + 1) if bucket_of(d) in needed]
    selected = np.flatnonzero(np.isin(dist, depths))
    if selected.size == 0:
        return

    selected_depth = dist[selected]
    for d in np.unique(selected_depth):
        indexes = selected[selected_depth == d]
        take = min(SAMPLES_PER_DEPTH, indexes.size)
        for i in rng.choice(indexes, size=take, replace=False):
            found.setdefault(key_of_state(states[int(i)].tolist()), int(d))


def generate_task(seed, needed, starts=()):
    rng = np.random.default_rng(seed)
    needed = set(needed)

    # Aim the hill climb at the deepest bucket that still needs boards.
    if rng.random() < DEEP_FRACTION:
        b = max(needed)
    else:
        b = int(rng.choice(sorted(needed)))
    lo, hi = BUCKET_RANGES[b]
    target = int(rng.integers(max(lo, 8), hi + 1))

    # Start from a stored deep board when there are any, else from a random one.
    state = result = None
    iterations = CLIMB_ITERATIONS
    if len(starts) and rng.random() < SEED_PROBABILITY:
        state = starts[int(rng.integers(len(starts)))]
        result = evaluate(state)
        if result is not None:
            # Wander around the deep plateau for a fixed number of steps.
            iterations = PLATEAU_ITERATIONS
            target = HI + 1
    while result is None:
        state = random_start(rng)
        result = evaluate(state)

    best = int(result[1].max())
    found = {}
    harvest(found, result, rng, needed)

    for _ in range(iterations):
        if best >= target or len(found) >= TASK_CAP:
            break
        candidate = mutate(state, rng)
        if candidate is None:
            continue
        candidate_result = evaluate(candidate)
        if candidate_result is None:
            continue
        depth = int(candidate_result[1].max())
        if depth >= best:
            # Equal-depth moves are harvested too: each one is a new, equally hard puzzle.
            state, result, best = candidate, candidate_result, depth
            harvest(found, result, rng, needed)

    return list(found.items())


# --------------------------------------------------------------- database
def load_game_matrices():

    matrices = set()

    folder = ROOT / "data" / "levels"

    if not folder.exists():
        raise FileNotFoundError(
            f"Game levels directory not found: {folder}"
        )

    for file in folder.glob("*.json"):
        with open(file) as f:
            for level in json.load(f):
                matrices.add("".join(np.array(level).ravel().astype(str)))

    return matrices


def initialize_database(db_path):

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS boards (
            board TEXT PRIMARY KEY,
            moves INTEGER,
            bucket INTEGER,
            accepted INTEGER NOT NULL,
            verified INTEGER NOT NULL DEFAULT 0
        )
    """)

    # Databases from earlier versions have no `verified` column.
    columns = [row[1] for row in conn.execute("PRAGMA table_info(boards)")]
    if "verified" not in columns:
        conn.execute("ALTER TABLE boards ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bucket
        ON boards(bucket, accepted)
    """)

    conn.commit()

    return conn


def get_counts(conn):

    rows = dict(conn.execute("""
        SELECT bucket, COUNT(*)
        FROM boards
        WHERE accepted = 1
        GROUP BY bucket
    """).fetchall())

    return [rows.get(i, 0) for i in range(N_BUCKETS)]


def report_progress(seen, counts, per_bucket, stats, start):

    elapsed = time.perf_counter() - start
    speed = seen / elapsed if elapsed else 0

    tqdm.write("\n" + "=" * 75)
    tqdm.write(f"Boards seen: {seen:,}  ({speed:,.1f} boards/sec)")
    tqdm.write(f"Collected: {sum(counts):,}/{per_bucket * N_BUCKETS:,}")
    for i, c in enumerate(counts):
        lo, hi = BUCKET_RANGES[i]
        tqdm.write(f"  bucket {i} (moves {lo}-{hi}): {c:,}/{per_bucket:,}")
    tqdm.write("Stats: " + " | ".join(f"{k}: {v:,}" for k, v in stats.items()))
    tqdm.write("=" * 75)


def data(
    attempts=ATTEMPTS,
    per_bucket=PER_BUCKET,
    db_path="slidebench.db",
    workers=WORKERS,
    batch_size=BATCH_SIZE,
    prior_dbs=(),
):

    print("\nLoading existing game matrices...")
    game_matrices = load_game_matrices()
    print(f"Loaded {len(game_matrices):,} existing matrices")

    conn = initialize_database(db_path)
    cur = conn.cursor()

    # Boards already stored by earlier runs are never generated again.
    prior_schemas = []
    for n, prior in enumerate(prior_dbs):
        if Path(prior).exists():
            schema = f"prior{n}"
            cur.execute(f"ATTACH DATABASE ? AS {schema}", (str(prior),))
            prior_schemas.append(schema)

    counts = get_counts(conn)

    print(f"Solution lengths: {LO} to {HI} in {N_BUCKETS} buckets")
    for i, (lo, hi) in enumerate(BUCKET_RANGES):
        print(f"  bucket {i}: moves {lo}-{hi}")
    print(f"Target per bucket: {per_bucket:,}  (total {per_bucket * N_BUCKETS:,})")
    print(f"Previously collected: {sum(counts):,}")
    print(f"Prior runs checked for duplicates: {len(prior_schemas)}")
    print(f"Workers: {workers}, batch size: {batch_size}, verify fraction: {VERIFY_FRACTION}")
    print(f"Database: {db_path}\n")

    rng = np.random.default_rng(SEED)
    pick = random.Random(SEED)

    stats = {
        "duplicates": 0,
        "existing": 0,
        "full_bucket": 0,
        "verified": 0,
        "label_mismatch": 0,
        "unsolved_or_too_deep": 0,
        "out_of_range": 0,
        "accepted": 0,
    }

    tables = ["boards"] + [f"{s}.boards" for s in prior_schemas]
    starts = []
    batch_no = 0

    start = time.perf_counter()
    seen = 0
    last_commit = 0
    next_report = REPORT_EVERY

    try:

        with ProcessPoolExecutor(max_workers=workers) as executor:

            with tqdm(
                total=per_bucket * N_BUCKETS,
                initial=sum(counts),
                desc="Accepted boards",
                unit="board",
            ) as pbar:

                while seen < attempts and not all(c >= per_bucket for c in counts):

                    needed = [b for b, c in enumerate(counts) if c < per_bucket]
                    seeds = rng.integers(0, 2 ** 62, size=batch_size)

                    # Reload the deep starting boards now and then; they improve as the DB fills.
                    if SEED_PROBABILITY > 0 and batch_no % SEED_REFRESH_EVERY == 0:
                        starts = load_seed_starts(cur, tables, max(max(needed) - 1, 0), SEED_POOL)
                    batch_no += 1

                    # 1. Cheap generation: the depth comes from the search that
                    #    produced the board, no extra BFS yet.
                    found = {}
                    total = 0

                    for result in executor.map(
                        generate_task, seeds, repeat(needed), repeat(starts), chunksize=1
                    ):
                        for key, depth in result:
                            total += 1
                            found.setdefault(key, depth)

                    stats["duplicates"] += total - len(found)
                    seen += total

                    # 2. Dedupe against this run's DB and every earlier run.
                    keys = list(found)
                    existing_db = set()

                    for i in range(0, len(keys), 900):
                        chunk = keys[i:i + 900]
                        placeholders = ",".join("?" for _ in chunk)
                        for table in tables:
                            existing_db.update(
                                row[0] for row in cur.execute(
                                    f"SELECT board FROM {table} WHERE board IN ({placeholders})",
                                    chunk
                                )
                            )

                    insert_rows = []
                    candidates = []

                    for key in keys:

                        if key in existing_db:
                            stats["duplicates"] += 1

                        elif key in game_matrices:
                            stats["existing"] += 1
                            insert_rows.append((key, None, None, 0, 0))

                        elif counts[bucket_of(found[key])] >= per_bucket:
                            # Stored as rejected so it is never processed again.
                            stats["full_bucket"] += 1
                            insert_rows.append((key, found[key], None, 0, 0))

                        else:
                            candidates.append(key)

                    # 3. Re-check only a random sample with the engine's BFS.
                    to_verify = [k for k in candidates if pick.random() < VERIFY_FRACTION]
                    verified = dict(zip(
                        to_verify,
                        executor.map(verify_key, to_verify, chunksize=8),
                    ))
                    stats["verified"] += len(verified)

                    # 4. Bucket and accept.
                    for key in candidates:

                        moves = found[key]
                        was_verified = key in verified

                        if was_verified:
                            if verified[key] != moves:
                                stats["label_mismatch"] += 1
                            moves = verified[key]

                        bucket = None
                        accepted = 0

                        if moves is None:
                            stats["unsolved_or_too_deep"] += 1

                        elif not LO <= moves <= HI:
                            stats["out_of_range"] += 1

                        else:
                            b = bucket_of(moves)
                            if counts[b] >= per_bucket:
                                stats["full_bucket"] += 1
                            else:
                                bucket = b
                                accepted = 1
                                counts[b] += 1
                                stats["accepted"] += 1

                        insert_rows.append((key, moves, bucket, accepted, int(was_verified)))

                    if insert_rows:
                        cur.executemany("""
                            INSERT OR IGNORE INTO boards
                            (board, moves, bucket, accepted, verified)
                            VALUES (?, ?, ?, ?, ?)
                        """, insert_rows)

                    pbar.n = sum(counts)
                    pbar.refresh()

                    if seen - last_commit >= COMMIT_EVERY:
                        conn.commit()
                        last_commit = seen

                    if seen >= next_report:
                        report_progress(seen, counts, per_bucket, stats, start)
                        next_report += REPORT_EVERY

        conn.commit()

    except KeyboardInterrupt:

        print("\nInterrupted! Saving database...")
        conn.commit()

    finally:

        conn.commit()

        elapsed = time.perf_counter() - start
        counts = get_counts(conn)
        total_rows = cur.execute("SELECT COUNT(*) FROM boards").fetchone()[0]

        print("\n" + "=" * 75)
        print("FINAL REPORT")
        print("=" * 75)
        print(f"Boards seen this run: {seen:,}")
        print(f"Total unique boards stored: {total_rows:,}")
        print(f"Accepted boards: {sum(counts):,}")
        print(f"Total time: {elapsed / 60:.2f} minutes")
        print("\nPer bucket:")

        for i, value in enumerate(counts):
            lo, hi = BUCKET_RANGES[i]
            print(f"  bucket {i} (moves {lo}-{hi}): {value:,}/{per_bucket:,}")

        print("\nRun statistics:")

        for key, value in stats.items():
            print(f"  {key}: {value:,}")

        if stats["label_mismatch"]:
            print("\nWARNING: verification BFS disagreed with evaluate() on some boards.")
            print("Check the engine rules against move_table()/expand() before trusting this data.")

        print("=" * 75)

        conn.close()

    return counts


if __name__ == "__main__":

    freeze_support()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    finished_dbs = []

    for run_id in range(1, TOTAL_RUNS + 1):

        print("\n" + "#" * 75)
        print(f"STARTING GENERATION RUN {run_id}/{TOTAL_RUNS}")
        print("#" * 75)

        run_dir = OUTPUT_DIR / f"run_{run_id}"
        db_path = run_dir / "slidebench.db"

        counts = data(db_path=db_path, prior_dbs=list(finished_dbs))

        finished_dbs.append(db_path)

        print(f"\nRun {run_id} completed: {db_path}")