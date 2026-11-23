"""Code prompts for drafter training data (MTP Head, Mon 11-02; for HPD-2 / Fable's E2, REPORT "HPD-1" notes): every
HumanEval+ problem is already used (80 in L1-20's training set, 24 held-out, 60 live benchmark prompts) and no other
code set is on disk, so these are new HumanEval-style function stubs (signature + docstring + doctests), written by hand
for this purpose and kept clear of HumanEval's 164 tasks. Rendered as EvalSuite / Leap render HumanEval+ (dfl2e_data.py
he_user: "Complete the following Python function. ..." in one user turn, thinking off); the 125B's answers are the data.
TASKS: list of (key, stub)."""

_S = [
("rle_encode", '''def rle_encode(s: str) -> str:
    """Run-length encode a string: every run of one repeated character becomes the character
    followed by the length of the run.
    >>> rle_encode("aaabcc")
    'a3b1c2'
    >>> rle_encode("")
    ''
    """
'''),
("rle_decode", '''def rle_decode(s: str) -> str:
    """Decode a run-length encoded string made of a character followed by a positive count
    (the count may have several digits).
    >>> rle_decode("a3b1c2")
    'aaabcc'
    >>> rle_decode("x12")
    'xxxxxxxxxxxx'
    """
'''),
("chunk_list", '''from typing import List


def chunk_list(items: List[int], size: int) -> List[List[int]]:
    """Split a list into consecutive chunks of the given size; the last chunk may be shorter.
    >>> chunk_list([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    >>> chunk_list([], 3)
    []
    """
'''),
("second_largest", '''from typing import List, Optional


def second_largest(nums: List[int]) -> Optional[int]:
    """Return the second largest distinct value in the list, or None if there is none.
    >>> second_largest([4, 1, 4, 3])
    3
    >>> second_largest([7, 7])
    """
'''),
("word_frequencies", '''from typing import Dict


def word_frequencies(text: str) -> Dict[str, int]:
    """Count how often each word occurs in the text, ignoring case and the punctuation
    characters .,!? attached to words.
    >>> word_frequencies("The cat and the hat.")
    {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}
    """
'''),
("is_isogram", '''def is_isogram(word: str) -> bool:
    """Return True if no letter appears more than once in the word, ignoring case,
    spaces and hyphens.
    >>> is_isogram("lumberjack")
    True
    >>> is_isogram("Alphabet")
    False
    >>> is_isogram("six-year-old")
    True
    """
'''),
("matrix_transpose", '''from typing import List


def matrix_transpose(m: List[List[int]]) -> List[List[int]]:
    """Return the transpose of a rectangular matrix given as a list of rows.
    >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    >>> matrix_transpose([])
    []
    """
'''),
("spiral_order", '''from typing import List


def spiral_order(m: List[List[int]]) -> List[int]:
    """Return the elements of a rectangular matrix in clockwise spiral order, starting at the
    top-left corner.
    >>> spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [1, 2, 3, 6, 9, 8, 7, 4, 5]
    >>> spiral_order([[1, 2], [3, 4], [5, 6]])
    [1, 2, 4, 6, 5, 3]
    """
'''),
("caesar_shift", '''def caesar_shift(text: str, k: int) -> str:
    """Shift every letter of the text k places forward in the alphabet, wrapping around and
    keeping case; other characters stay as they are. k may be negative.
    >>> caesar_shift("Hello, World!", 3)
    'Khoor, Zruog!'
    >>> caesar_shift("abc", -1)
    'zab'
    """
'''),
("balanced_brackets", '''def balanced_brackets(s: str) -> bool:
    """Return True if every bracket in s, of the kinds (), [] and {}, is closed by the matching
    kind in the right order. Other characters are ignored.
    >>> balanced_brackets("a(b[c]{d})")
    True
    >>> balanced_brackets("(]")
    False
    >>> balanced_brackets("((")
    False
    """
'''),
("merge_intervals", '''from typing import List, Tuple


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping closed intervals and return them sorted by start. Intervals that only
    touch (end == next start) are merged too.
    >>> merge_intervals([(1, 3), (2, 6), (8, 10), (10, 12)])
    [(1, 6), (8, 12)]
    >>> merge_intervals([])
    []
    """
'''),
("binary_search_left", '''from typing import List


def binary_search_left(a: List[int], x: int) -> int:
    """Given a sorted list, return the first index i such that a[i] >= x (len(a) if none),
    using binary search.
    >>> binary_search_left([1, 2, 2, 4], 2)
    1
    >>> binary_search_left([1, 2, 2, 4], 5)
    4
    """
'''),
("digital_root", '''def digital_root(n: int) -> int:
    """Repeatedly sum the decimal digits of a non-negative integer until one digit remains.
    >>> digital_root(493193)
    2
    >>> digital_root(0)
    0
    """
'''),
("collatz_steps", '''def collatz_steps(n: int) -> int:
    """Return how many steps the Collatz process (n -> n / 2 if even, 3n + 1 if odd) needs to
    reach 1 from a positive integer n.
    >>> collatz_steps(1)
    0
    >>> collatz_steps(6)
    8
    """
'''),
("pascal_row", '''from typing import List


def pascal_row(n: int) -> List[int]:
    """Return row n of Pascal's triangle (row 0 is [1]).
    >>> pascal_row(0)
    [1]
    >>> pascal_row(4)
    [1, 4, 6, 4, 1]
    """
'''),
("to_snake_case", '''def to_snake_case(name: str) -> str:
    """Convert a camelCase or PascalCase identifier to snake_case. Runs of capitals are treated
    as one word, e.g. an acronym.
    >>> to_snake_case("parseHttpResponse")
    'parse_http_response'
    >>> to_snake_case("XMLParser")
    'xml_parser'
    """
'''),
("to_camel_case", '''def to_camel_case(name: str) -> str:
    """Convert a snake_case identifier to camelCase; leading and repeated underscores are
    dropped.
    >>> to_camel_case("read_file_lines")
    'readFileLines'
    >>> to_camel_case("__init")
    'init'
    """
'''),
("roman_to_int", '''def roman_to_int(s: str) -> int:
    """Convert a Roman numeral in standard subtractive form (I, V, X, L, C, D, M) to an integer.
    >>> roman_to_int("MCMXCIV")
    1994
    >>> roman_to_int("III")
    3
    """
'''),
("flatten_nested", '''from typing import Any, List


def flatten_nested(x: List[Any]) -> List[Any]:
    """Flatten arbitrarily nested lists into one list, keeping the order of the leaves.
    >>> flatten_nested([1, [2, [3, 4]], [], 5])
    [1, 2, 3, 4, 5]
    """
'''),
("group_anagrams", '''from typing import List


def group_anagrams(words: List[str]) -> List[List[str]]:
    """Group words that are anagrams of each other. Groups keep the order of first appearance,
    and words inside a group keep their input order.
    >>> group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
    """
'''),
("longest_common_prefix", '''from typing import List


def longest_common_prefix(words: List[str]) -> str:
    """Return the longest string that is a prefix of every word ('' for an empty list).
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar"])
    ''
    """
'''),
("count_islands", '''from typing import List


def count_islands(grid: List[List[int]]) -> int:
    """Count groups of 1-cells connected horizontally or vertically in a grid of 0s and 1s.
    >>> count_islands([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    3
    >>> count_islands([])
    0
    """
'''),
("knapsack_max_value", '''from typing import List


def knapsack_max_value(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 knapsack: the largest total value of a subset of items whose total weight is at most
    capacity.
    >>> knapsack_max_value([1, 3, 4], [15, 20, 30], 4)
    35
    >>> knapsack_max_value([5], [10], 4)
    0
    """
'''),
("edit_distance", '''def edit_distance(a: str, b: str) -> int:
    """Return the Levenshtein distance between two strings: the fewest single-character
    insertions, deletions or substitutions that turn a into b.
    >>> edit_distance("kitten", "sitting")
    3
    >>> edit_distance("", "abc")
    3
    """
'''),
("lis_length", '''from typing import List


def lis_length(nums: List[int]) -> int:
    """Length of the longest strictly increasing subsequence.
    >>> lis_length([10, 9, 2, 5, 3, 7, 101, 18])
    4
    >>> lis_length([])
    0
    """
'''),
("coin_change_ways", '''from typing import List


def coin_change_ways(coins: List[int], amount: int) -> int:
    """Number of ways to make amount from unlimited coins of the given values (order of coins
    does not matter).
    >>> coin_change_ways([1, 2, 5], 5)
    4
    >>> coin_change_ways([2], 3)
    0
    """
'''),
("min_coins", '''from typing import List


def min_coins(coins: List[int], amount: int) -> int:
    """Fewest coins (unlimited supply of each value) that sum to amount, or -1 if impossible.
    >>> min_coins([1, 2, 5], 11)
    3
    >>> min_coins([2], 3)
    -1
    """
'''),
("rotate_list", '''from typing import List


def rotate_list(items: List[int], k: int) -> List[int]:
    """Rotate the list k places to the right (k may exceed the length or be negative).
    >>> rotate_list([1, 2, 3, 4, 5], 2)
    [4, 5, 1, 2, 3]
    >>> rotate_list([1, 2, 3], -1)
    [2, 3, 1]
    """
'''),
("kth_smallest", '''from typing import List


def kth_smallest(nums: List[int], k: int) -> int:
    """Return the k-th smallest element (1-based) of an unsorted list with duplicates counted.
    >>> kth_smallest([7, 10, 4, 3, 20, 15], 3)
    7
    >>> kth_smallest([2, 2, 1], 2)
    2
    """
'''),
("valid_ipv4", '''def valid_ipv4(s: str) -> bool:
    """Return True if s is a dotted IPv4 address: four decimal numbers 0-255 without leading
    zeros (a single 0 is fine), separated by dots.
    >>> valid_ipv4("192.168.0.1")
    True
    >>> valid_ipv4("256.1.1.1")
    False
    >>> valid_ipv4("01.2.3.4")
    False
    """
'''),
("parse_duration", '''def parse_duration(s: str) -> int:
    """Parse a duration such as "1h30m", "45s" or "2h5s" (hours h, minutes m, seconds s, each at
    most once, in that order) and return the total number of seconds.
    >>> parse_duration("1h30m")
    5400
    >>> parse_duration("2h5s")
    7205
    """
'''),
("format_seconds", '''def format_seconds(t: int) -> str:
    """Format a non-negative number of seconds as H:MM:SS.
    >>> format_seconds(3725)
    '1:02:05'
    >>> format_seconds(59)
    '0:00:59'
    """
'''),
("days_between", '''def days_between(a: str, b: str) -> int:
    """Number of days from date a to date b, both 'YYYY-MM-DD' (negative if b is earlier),
    without using the datetime module. Use the Gregorian leap-year rules.
    >>> days_between("2024-02-27", "2024-03-01")
    3
    >>> days_between("2023-01-01", "2022-12-31")
    -1
    """
'''),
("day_of_week", '''def day_of_week(y: int, m: int, d: int) -> str:
    """Return the English weekday name of a Gregorian date (e.g. with Zeller's congruence).
    >>> day_of_week(2024, 1, 1)
    'Monday'
    >>> day_of_week(2000, 2, 29)
    'Tuesday'
    """
'''),
("matrix_multiply", '''from typing import List


def matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """Multiply two matrices given as lists of rows (the number of columns of a equals the
    number of rows of b).
    >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]
    """
'''),
("determinant", '''from typing import List


def determinant(m: List[List[float]]) -> float:
    """Determinant of a square matrix by Gaussian elimination with partial pivoting.
    >>> round(determinant([[2, 0], [1, 3]]), 6)
    6.0
    >>> round(determinant([[1, 2], [2, 4]]), 6)
    0.0
    """
'''),
("sieve_primes", '''from typing import List


def sieve_primes(n: int) -> List[int]:
    """All primes up to and including n, by the sieve of Eratosthenes.
    >>> sieve_primes(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
    >>> sieve_primes(1)
    []
    """
'''),
("prime_factor_counts", '''from typing import Dict


def prime_factor_counts(n: int) -> Dict[int, int]:
    """Map each prime factor of n (n >= 2) to its exponent.
    >>> prime_factor_counts(360)
    {2: 3, 3: 2, 5: 1}
    >>> prime_factor_counts(13)
    {13: 1}
    """
'''),
("lcm_list", '''from typing import List


def lcm_list(nums: List[int]) -> int:
    """Least common multiple of a non-empty list of positive integers.
    >>> lcm_list([4, 6, 10])
    60
    >>> lcm_list([7])
    7
    """
'''),
("perfect_numbers", '''from typing import List


def perfect_numbers(limit: int) -> List[int]:
    """Perfect numbers (equal to the sum of their proper divisors) up to limit.
    >>> perfect_numbers(500)
    [6, 28, 496]
    """
'''),
("armstrong_numbers", '''from typing import List


def armstrong_numbers(lo: int, hi: int) -> List[int]:
    """Numbers in [lo, hi] equal to the sum of their digits each raised to the number of digits.
    >>> armstrong_numbers(100, 400)
    [153, 370, 371]
    """
'''),
("int_to_words", '''def int_to_words(n: int) -> str:
    """Spell a number 0 <= n < 1000 in English words, using hyphens for 21-99.
    >>> int_to_words(0)
    'zero'
    >>> int_to_words(342)
    'three hundred forty-two'
    >>> int_to_words(15)
    'fifteen'
    """
'''),
("base_convert", '''def base_convert(digits: str, src: int, dst: int) -> str:
    """Convert a non-negative number written in base src (2-36, digits 0-9 then lowercase a-z)
    to base dst.
    >>> base_convert("ff", 16, 2)
    '11111111'
    >>> base_convert("0", 10, 7)
    '0'
    """
'''),
("bit_count_range", '''from typing import List


def bit_count_range(n: int) -> List[int]:
    """For every i in 0..n return the number of 1 bits in i, computed from earlier results.
    >>> bit_count_range(5)
    [0, 1, 1, 2, 1, 2]
    """
'''),
("gray_code", '''from typing import List


def gray_code(n: int) -> List[int]:
    """The n-bit reflected Gray code sequence, starting at 0.
    >>> gray_code(2)
    [0, 1, 3, 2]
    >>> gray_code(0)
    [0]
    """
'''),
("power_set", '''from typing import List


def power_set(items: List[int]) -> List[List[int]]:
    """All subsets of the list, in the order produced by adding each item to every subset
    built so far (starting from the empty subset).
    >>> power_set([1, 2])
    [[], [1], [2], [1, 2]]
    """
'''),
("permutations_of", '''from typing import List


def permutations_of(s: str) -> List[str]:
    """All distinct permutations of the characters of s, sorted.
    >>> permutations_of("aba")
    ['aab', 'aba', 'baa']
    """
'''),
("combinations_sum", '''from typing import List


def combinations_sum(candidates: List[int], target: int) -> List[List[int]]:
    """All combinations of the distinct positive candidates (each usable any number of times)
    that sum to target; each combination non-decreasing, the list sorted.
    >>> combinations_sum([2, 3, 6, 7], 7)
    [[2, 2, 3], [7]]
    """
'''),
("n_queens_count", '''def n_queens_count(n: int) -> int:
    """Number of ways to place n queens on an n x n board so that none attack each other.
    >>> n_queens_count(4)
    2
    >>> n_queens_count(1)
    1
    """
'''),
("sudoku_valid", '''from typing import List


def sudoku_valid(board: List[List[int]]) -> bool:
    """Check a partly filled 9 x 9 Sudoku board (0 = empty): no digit repeats in any row,
    column or 3 x 3 box. Empty cells are allowed.
    """
'''),
("tic_tac_toe_winner", '''from typing import List


def tic_tac_toe_winner(board: List[str]) -> str:
    """Given three strings of three characters each ('X', 'O' or '.'), return 'X' or 'O' if that
    player has three in a row, column or diagonal, otherwise '.'.
    >>> tic_tac_toe_winner(["XOX", ".XO", "O.X"])
    'X'
    >>> tic_tac_toe_winner(["XO.", "...", "..."])
    '.'
    """
'''),
("shortest_path_grid", '''from typing import List


def shortest_path_grid(grid: List[str]) -> int:
    """grid rows contain '.', '#', 'S' and 'E'. Return the fewest moves (up / down / left / right,
    never into '#') from S to E, or -1 if E cannot be reached.
    >>> shortest_path_grid(["S.#", "..#", "#.E"])
    4
    """
'''),
("topological_order", '''from typing import Dict, List


def topological_order(deps: Dict[str, List[str]]) -> List[str]:
    """deps maps each task to the tasks it depends on. Return an order in which every task comes
    after its dependencies, choosing the alphabetically smallest available task at each step;
    raise ValueError on a cycle. Tasks that only appear as dependencies are included.
    >>> topological_order({"b": ["a"], "c": ["a", "b"]})
    ['a', 'b', 'c']
    """
'''),
("dijkstra", '''from typing import Dict, List, Tuple


def dijkstra(graph: Dict[str, List[Tuple[str, int]]], start: str) -> Dict[str, int]:
    """Shortest distances from start to every reachable node of a graph with non-negative edge
    weights, given as adjacency lists of (neighbour, weight).
    >>> dijkstra({"a": [("b", 1), ("c", 4)], "b": [("c", 2)], "c": []}, "a")
    {'a': 0, 'b': 1, 'c': 3}
    """
'''),
("connected_components", '''from typing import List, Tuple


def connected_components(n: int, edges: List[Tuple[int, int]]) -> int:
    """Number of connected components of an undirected graph with nodes 0..n-1, using
    union-find.
    >>> connected_components(5, [(0, 1), (1, 2), (3, 4)])
    2
    """
'''),
("bipartite", '''from typing import List


def bipartite(adj: List[List[int]]) -> bool:
    """Return True if the undirected graph given as adjacency lists can be two-coloured.
    >>> bipartite([[1, 3], [0, 2], [1, 3], [0, 2]])
    True
    >>> bipartite([[1, 2], [0, 2], [0, 1]])
    False
    """
'''),
("lru_cache_sim", '''from typing import List


def lru_cache_sim(capacity: int, requests: List[int]) -> int:
    """Simulate an LRU cache of the given capacity over a sequence of requested keys and return
    the number of misses.
    >>> lru_cache_sim(2, [1, 2, 1, 3, 2, 1])
    5
    """
'''),
("moving_average", '''from typing import List


def moving_average(xs: List[float], k: int) -> List[float]:
    """Averages of every window of k consecutive values (an empty list if k > len(xs)).
    >>> moving_average([1, 2, 3, 4, 5], 3)
    [2.0, 3.0, 4.0]
    """
'''),
("running_median", '''from typing import List


def running_median(xs: List[int]) -> List[float]:
    """The median of the first i values for every i = 1..len(xs), using two heaps.
    >>> running_median([5, 15, 1, 3])
    [5.0, 10.0, 5.0, 4.0]
    """
'''),
("stock_max_profit", '''from typing import List


def stock_max_profit(prices: List[int]) -> int:
    """Largest profit from one buy followed by one later sell (0 if prices only fall).
    >>> stock_max_profit([7, 1, 5, 3, 6, 4])
    5
    >>> stock_max_profit([7, 6, 4])
    0
    """
'''),
("max_subarray", '''from typing import List, Tuple


def max_subarray(nums: List[int]) -> Tuple[int, int, int]:
    """Kadane's algorithm: return (best sum, start index, end index) of the non-empty contiguous
    subarray with the largest sum; on ties keep the earliest start.
    >>> max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    (6, 3, 6)
    """
'''),
("trap_rain_water", '''from typing import List


def trap_rain_water(h: List[int]) -> int:
    """Units of water trapped between bars of the given heights after rain.
    >>> trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
    6
    """
'''),
("two_sum_indices", '''from typing import List, Tuple


def two_sum_indices(nums: List[int], target: int) -> Tuple[int, int]:
    """Indices i < j with nums[i] + nums[j] == target, the pair with the smallest j (then the
    smallest i); (-1, -1) if none. One pass with a dictionary.
    >>> two_sum_indices([2, 7, 11, 15], 9)
    (0, 1)
    """
'''),
("three_sum_triplets", '''from typing import List


def three_sum_triplets(nums: List[int]) -> List[List[int]]:
    """All distinct triplets (sorted inside, list sorted) of values from nums summing to zero.
    >>> three_sum_triplets([-1, 0, 1, 2, -1, -4])
    [[-1, -1, 2], [-1, 0, 1]]
    """
'''),
("majority_element", '''from typing import List, Optional


def majority_element(nums: List[int]) -> Optional[int]:
    """The value occurring more than len(nums) // 2 times (Boyer-Moore vote, then a check), or
    None.
    >>> majority_element([2, 2, 1, 1, 2])
    2
    >>> majority_element([1, 2])
    """
'''),
("dutch_flag_sort", '''from typing import List


def dutch_flag_sort(nums: List[int]) -> List[int]:
    """Sort a list containing only 0, 1 and 2 in one pass with three pointers and return it.
    >>> dutch_flag_sort([2, 0, 2, 1, 1, 0])
    [0, 0, 1, 1, 2, 2]
    """
'''),
("merge_sorted_lists", '''from typing import List


def merge_sorted_lists(a: List[int], b: List[int]) -> List[int]:
    """Merge two sorted lists into one sorted list without calling sort.
    >>> merge_sorted_lists([1, 4, 9], [2, 3, 10])
    [1, 2, 3, 4, 9, 10]
    """
'''),
("merge_sort", '''from typing import List


def merge_sort(xs: List[int]) -> List[int]:
    """Return a sorted copy of xs using top-down merge sort.
    >>> merge_sort([5, 2, 4, 6, 1, 3])
    [1, 2, 3, 4, 5, 6]
    """
'''),
("quickselect_median", '''from typing import List


def quickselect_median(xs: List[int]) -> float:
    """Median of a non-empty list using quickselect (the mean of the two middle values for an
    even length).
    >>> quickselect_median([3, 1, 2])
    2.0
    >>> quickselect_median([4, 1, 3, 2])
    2.5
    """
'''),
("counting_inversions", '''from typing import List


def counting_inversions(xs: List[int]) -> int:
    """Number of pairs i < j with xs[i] > xs[j], in O(n log n).
    >>> counting_inversions([2, 4, 1, 3, 5])
    3
    """
'''),
("next_permutation", '''from typing import List


def next_permutation(xs: List[int]) -> List[int]:
    """The next lexicographic permutation of xs (the smallest one if xs is the largest).
    >>> next_permutation([1, 2, 3])
    [1, 3, 2]
    >>> next_permutation([3, 2, 1])
    [1, 2, 3]
    """
'''),
("look_and_say", '''def look_and_say(n: int) -> str:
    """The n-th term (1-based) of the look-and-say sequence starting from '1'.
    >>> look_and_say(1)
    '1'
    >>> look_and_say(5)
    '111221'
    """
'''),
("zigzag_convert", '''def zigzag_convert(s: str, rows: int) -> str:
    """Write s in a zigzag over the given number of rows and read it row by row.
    >>> zigzag_convert("PAYPALISHIRING", 3)
    'PAHNAPLSIIGYIR'
    """
'''),
("text_justify_line", '''from typing import List


def text_justify_line(words: List[str], width: int) -> str:
    """Join words into one line of exactly width characters, spreading the extra spaces as
    evenly as possible and giving the leftmost gaps one more when they do not divide evenly. A
    single word is padded on the right.
    >>> text_justify_line(["This", "is", "an"], 16)
    'This    is    an'
    """
'''),
("wrap_text", '''from typing import List


def wrap_text(text: str, width: int) -> List[str]:
    """Greedy word wrap: split text into lines of at most width characters, breaking only at
    spaces (a longer word stays alone on its line).
    >>> wrap_text("the quick brown fox jumps", 10)
    ['the quick', 'brown fox', 'jumps']
    """
'''),
("strip_comments", '''def strip_comments(code: str) -> str:
    """Remove Python-style comments (from # to the end of the line) from code, ignoring # inside
    single- or double-quoted strings, and strip the trailing spaces of each line.
    >>> strip_comments("x = 1  # one\\ns = '#no'  # yes")
    "x = 1\\ns = '#no'"
    """
'''),
("tokenize_arith", '''from typing import List


def tokenize_arith(expr: str) -> List[str]:
    """Split an arithmetic expression into tokens: integers, decimal numbers, the operators
    + - * / ^ and parentheses. Spaces are skipped.
    >>> tokenize_arith("3 + 4.5*(2 - 1)")
    ['3', '+', '4.5', '*', '(', '2', '-', '1', ')']
    """
'''),
("eval_rpn", '''from typing import List


def eval_rpn(tokens: List[str]) -> int:
    """Evaluate an expression in reverse Polish notation with integers and + - * /, where /
    truncates toward zero.
    >>> eval_rpn(["2", "1", "+", "3", "*"])
    9
    >>> eval_rpn(["4", "13", "5", "/", "+"])
    6
    """
'''),
("infix_to_postfix", '''from typing import List


def infix_to_postfix(tokens: List[str]) -> List[str]:
    """Shunting-yard: convert infix tokens (numbers, + - * /, parentheses) to postfix, with * and
    / binding tighter than + and -, all left-associative.
    >>> infix_to_postfix(["3", "+", "4", "*", "2"])
    ['3', '4', '2', '*', '+']
    """
'''),
("eval_expression", '''def eval_expression(expr: str) -> float:
    """Evaluate an arithmetic expression with + - * /, parentheses, unary minus and decimal
    numbers without using eval.
    >>> eval_expression("2 * (3 + 4) - -1")
    15.0
    """
'''),
("parse_csv_line", '''from typing import List


def parse_csv_line(line: str) -> List[str]:
    """Split one CSV line on commas; fields may be enclosed in double quotes, inside which commas
    are literal and "" stands for one quote.
    >>> parse_csv_line('a,"b,c",d')
    ['a', 'b,c', 'd']
    >>> parse_csv_line('"say ""hi""",x')
    ['say "hi"', 'x']
    """
'''),
("parse_query_string", '''from typing import Dict, List


def parse_query_string(q: str) -> Dict[str, List[str]]:
    """Parse a URL query string such as 'a=1&b=2&a=3' into a dict of value lists, decoding '+'
    as a space and %XX escapes. Keys without '=' get an empty string value.
    >>> parse_query_string("a=1&b=x+y&a=3&flag")
    {'a': ['1', '3'], 'b': ['x y'], 'flag': ['']}
    """
'''),
("ini_parse", '''from typing import Dict


def ini_parse(text: str) -> Dict[str, Dict[str, str]]:
    """Parse INI text: [section] headers, key = value lines (spaces around both stripped),
    blank lines and lines starting with ; or # ignored. Keys before any section go to ''.
    >>> ini_parse("[db]\\nhost = x\\n; note\\nport=5")
    {'db': {'host': 'x', 'port': '5'}}
    """
'''),
("json_path_get", '''from typing import Any


def json_path_get(obj: Any, path: str) -> Any:
    """Follow a dotted path through nested dicts and lists; numeric parts index lists. Return
    None if any step is missing.
    >>> json_path_get({"a": {"b": [10, {"c": 3}]}}, "a.b.1.c")
    3
    >>> json_path_get({"a": 1}, "a.x")
    """
'''),
("deep_merge", '''from typing import Any, Dict


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with b merged into a: nested dicts merge recursively, every other value
    from b replaces the one in a. The inputs are not modified.
    >>> deep_merge({"x": {"y": 1, "z": 2}, "k": 0}, {"x": {"z": 3}, "k": [1]})
    {'x': {'y': 1, 'z': 3}, 'k': [1]}
    """
'''),
("invert_dict", '''from typing import Dict, List


def invert_dict(d: Dict[str, int]) -> Dict[int, List[str]]:
    """Map each value to the sorted list of keys that had it.
    >>> invert_dict({"a": 1, "b": 2, "c": 1})
    {1: ['a', 'c'], 2: ['b']}
    """
'''),
("top_k_frequent", '''from typing import List


def top_k_frequent(words: List[str], k: int) -> List[str]:
    """The k most frequent words, most frequent first; ties broken alphabetically.
    >>> top_k_frequent(["a", "b", "a", "c", "b", "a"], 2)
    ['a', 'b']
    """
'''),
("first_unique_char", '''def first_unique_char(s: str) -> int:
    """Index of the first character that occurs exactly once in s, or -1.
    >>> first_unique_char("leetcode")
    0
    >>> first_unique_char("aabb")
    -1
    """
'''),
("longest_unique_substring", '''def longest_unique_substring(s: str) -> str:
    """The longest substring without repeated characters (the first one on ties), with a sliding
    window.
    >>> longest_unique_substring("abcabcbb")
    'abc'
    >>> longest_unique_substring("pwwkew")
    'wke'
    """
'''),
("min_window_substring", '''def min_window_substring(s: str, t: str) -> str:
    """The shortest substring of s containing every character of t with multiplicity (the first
    one on ties), or '' if there is none.
    >>> min_window_substring("ADOBECODEBANC", "ABC")
    'BANC'
    """
'''),
("longest_palindromic_substring", '''def longest_palindromic_substring(s: str) -> str:
    """The longest contiguous palindrome in s (the first one on ties), by expanding around
    centres.
    >>> longest_palindromic_substring("babad")
    'bab'
    >>> longest_palindromic_substring("cbbd")
    'bb'
    """
'''),
("is_subsequence", '''def is_subsequence(small: str, big: str) -> bool:
    """Return True if small can be obtained from big by deleting characters.
    >>> is_subsequence("ace", "abcde")
    True
    >>> is_subsequence("aec", "abcde")
    False
    """
'''),
("lcs_string", '''def lcs_string(a: str, b: str) -> str:
    """One longest common subsequence of a and b, reconstructed from the dynamic-programming
    table (prefer stepping in a when both directions tie).
    >>> len(lcs_string("ABCBDAB", "BDCABA"))
    4
    """
'''),
("wildcard_match", '''def wildcard_match(s: str, p: str) -> bool:
    """Match s against pattern p where ? matches one character and * any sequence (also empty);
    the whole string must match.
    >>> wildcard_match("adceb", "*a*b")
    True
    >>> wildcard_match("acdcb", "a*c?b")
    False
    """
'''),
("word_break", '''from typing import List


def word_break(s: str, words: List[str]) -> bool:
    """Return True if s can be split into a sequence of the given dictionary words (each usable
    many times).
    >>> word_break("applepenapple", ["apple", "pen"])
    True
    >>> word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
    False
    """
'''),
("decode_ways", '''def decode_ways(s: str) -> int:
    """Number of ways to decode a digit string where 'A' = 1 ... 'Z' = 26.
    >>> decode_ways("226")
    3
    >>> decode_ways("06")
    0
    """
'''),
("climb_stairs", '''def climb_stairs(n: int, steps: tuple = (1, 2)) -> int:
    """Number of distinct ways to climb n stairs taking any of the allowed step sizes each move.
    >>> climb_stairs(4)
    5
    >>> climb_stairs(4, (1, 3))
    3
    """
'''),
("unique_grid_paths", '''from typing import List


def unique_grid_paths(grid: List[List[int]]) -> int:
    """Number of paths from the top-left to the bottom-right cell moving only right or down,
    avoiding cells that are 1.
    >>> unique_grid_paths([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    2
    """
'''),
("house_robber", '''from typing import List


def house_robber(values: List[int]) -> int:
    """Largest sum of values with no two adjacent elements chosen.
    >>> house_robber([2, 7, 9, 3, 1])
    12
    """
'''),
("partition_equal_sum", '''from typing import List


def partition_equal_sum(nums: List[int]) -> bool:
    """Return True if the positive integers can be split into two groups with equal sums.
    >>> partition_equal_sum([1, 5, 11, 5])
    True
    >>> partition_equal_sum([1, 2, 3, 5])
    False
    """
'''),
("histogram_largest_rect", '''from typing import List


def histogram_largest_rect(h: List[int]) -> int:
    """Area of the largest rectangle under a histogram of unit-width bars, with a stack.
    >>> histogram_largest_rect([2, 1, 5, 6, 2, 3])
    10
    """
'''),
("daily_warmer", '''from typing import List


def daily_warmer(temps: List[int]) -> List[int]:
    """For each day, how many days until a strictly warmer one (0 if never), with a monotonic
    stack.
    >>> daily_warmer([73, 74, 75, 71, 69, 72, 76, 73])
    [1, 1, 4, 2, 1, 1, 0, 0]
    """
'''),
]

# Too close to one of the 24 held-out (dfl2d_code) or the 60 live-benchmark (dec_humaneval) HumanEval tasks, so left out:
# held-out 96 count_up_to, 40 triples_sum_to_zero, 47 median, 59 largest_prime_factor, 132 is_nested, 10 make_palindrome;
# live 114 minSubArraySum, 156 int_to_mini_roman, 79 decimal_to_binary, 93 encode, 56 correct_bracketing.
DROP = {"sieve_primes", "three_sum_triplets", "quickselect_median", "prime_factor_counts", "balanced_brackets",
        "longest_palindromic_substring", "max_subarray", "roman_to_int", "base_convert", "caesar_shift"}
assert DROP <= {n for n, _ in _S}
TASKS = [(f"ct_{name}", stub) for name, stub in _S if name not in DROP]
assert len({k for k, _ in TASKS}) == len(TASKS)
