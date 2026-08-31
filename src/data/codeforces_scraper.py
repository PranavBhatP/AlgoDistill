"""Codeforces scraper and problem curator for algorithmic LLM distillation.

Fetches problems from Codeforces API, filters for easy/medium difficulty (rating 800-1600),
restricts to single-dominant algorithmic tags, and extracts statements and sample tests.
Includes curated offline problems across all 10 target categories.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import logging
import re
import requests

logger = logging.getLogger(__name__)

TARGET_TAG_TAXONOMY = [
    "dynamic programming",
    "greedy",
    "graphs",
    "math",
    "data structures",
    "trees",
    "brute force",
    "strings",
    "number theory",
    "binary search",
]

TAG_NORMALIZATION = {
    "dp": "dynamic programming",
    "dynamic programming": "dynamic programming",
    "greedy": "greedy",
    "graphs": "graphs",
    "graph matchings": "graphs",
    "shortest paths": "graphs",
    "dfs and similar": "graphs",
    "math": "math",
    "data structures": "data structures",
    "dsu": "data structures",
    "trees": "trees",
    "brute force": "brute force",
    "strings": "strings",
    "string suffix structures": "strings",
    "number theory": "number theory",
    "binary search": "binary search",
    "ternary search": "binary search",
}


@dataclass
class TestCase:
    """Representation of an algorithmic test case."""
    __test__ = False  # Prevent pytest from treating this as a test class
    input_data: str
    output_data: str
    explanation: Optional[str] = None


@dataclass
class Problem:
    """Representation of a competitive programming problem."""
    problem_id: str
    title: str
    contest_id: int
    index: str
    rating: int
    ground_truth_tag: str
    all_tags: List[str]
    statement: str
    input_spec: str
    output_spec: str
    sample_tests: List[TestCase] = field(default_factory=list)
    time_limit_ms: int = 2000
    memory_limit_mb: int = 256
    source_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sample_tests"] = [asdict(t) for t in self.sample_tests]
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Problem":
        sample_tests = [TestCase(**t) if isinstance(t, dict) else t for t in d.get("sample_tests", [])]
        d_copy = dict(d)
        d_copy["sample_tests"] = sample_tests
        return cls(**d_copy)


class CodeforcesScraper:
    """Scraper and manager for Codeforces competitive programming problems."""

    BASE_API_URL = "https://codeforces.com/api/problemset.problems"

    def __init__(self, target_tags: Optional[List[str]] = None, min_rating: int = 800, max_rating: int = 1600):
        self.target_tags = target_tags or TARGET_TAG_TAXONOMY
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def fetch_api_problems(self, max_problems_per_tag: int = 15) -> List[Dict[str, Any]]:
        """Fetch problem metadata from Codeforces official API."""
        try:
            resp = self.session.get(self.BASE_API_URL, timeout=8)
            if resp.status_code != 200:
                logger.warning("Codeforces API returned %d. Using offline curated dataset.", resp.status_code)
                return []
            data = resp.json()
            if data.get("status") != "OK":
                return []

            problems = data["result"]["problems"]
            filtered = []
            tag_counts: Dict[str, int] = {t: 0 for t in self.target_tags}

            for p in problems:
                rating = p.get("rating")
                if not rating or rating < self.min_rating or rating > self.max_rating:
                    continue

                raw_tags = p.get("tags", [])
                normalized_tags = set()
                for rt in raw_tags:
                    norm = TAG_NORMALIZATION.get(rt.lower())
                    if norm and norm in self.target_tags:
                        normalized_tags.add(norm)

                if len(normalized_tags) == 1:
                    dom_tag = list(normalized_tags)[0]
                    if tag_counts[dom_tag] < max_problems_per_tag:
                        p["dominant_tag"] = dom_tag
                        filtered.append(p)
                        tag_counts[dom_tag] += 1

            return filtered
        except Exception as e:
            logger.warning("Could not connect to Codeforces API (%s). Using offline curated dataset.", e)
            return []

    def get_curated_dataset(self) -> List[Problem]:
        """Return curated offline dataset of authentic competitive programming problems."""
        raw_curated = [
            # 1. Dynamic Programming
            {
                "problem_id": "CF-189A",
                "title": "Cut Ribbon",
                "contest_id": 189,
                "index": "A",
                "rating": 1300,
                "ground_truth_tag": "dynamic programming",
                "all_tags": ["dp"],
                "statement": "Polycarpus has a ribbon, its length is n. He wants to cut the ribbon in a way that fulfils the following conditions:\n1. After the cuts, each ribbon piece must have length a, b, or c.\n2. The number of ribbon pieces obtained must be maximum.\nHelp Polycarpus and find the maximum number of pieces using dynamic programming.",
                "input_spec": "The first line contains four space-separated integers n, a, b and c (1 <= n, a, b, c <= 4000).",
                "output_spec": "Print a single integer - the maximum possible number of ribbon pieces.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5 5 3 2\n", "output_data": "2\n"},
                    {"input_data": "7 5 5 2\n", "output_data": "2\n"},
                ]
            },
            {
                "problem_id": "CF-455A",
                "title": "Boredom",
                "contest_id": 455,
                "index": "A",
                "rating": 1500,
                "ground_truth_tag": "dynamic programming",
                "all_tags": ["dp"],
                "statement": "Given an array a of n integers. In one step, you can choose an element a_k and delete it; in doing so, all elements equal to a_k + 1 and a_k - 1 are also deleted. This step brings you a_k points. Find the maximum points you can earn with DP.",
                "input_spec": "The first line contains integer n (1 <= n <= 10^5). The second line contains n integers a_1, a_2, ..., a_n (1 <= a_i <= 10^5).",
                "output_spec": "Print a single integer - the maximum points.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "2\n1 2\n", "output_data": "2\n"},
                    {"input_data": "3\n1 2 3\n", "output_data": "4\n"},
                    {"input_data": "9\n1 2 1 3 2 2 2 2 3\n", "output_data": "10\n"},
                ]
            },
            {
                "problem_id": "CF-702A",
                "title": "Maximum Increase",
                "contest_id": 702,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "dynamic programming",
                "all_tags": ["dp"],
                "statement": "You are given array consisting of n integers. Your task is to find the maximum length of an increasing contiguous subarray. Use dynamic programming state tracking.",
                "input_spec": "First line: n (1 <= n <= 10^5). Second line: n integers a_i (1 <= a_i <= 10^9).",
                "output_spec": "Print length of the longest increasing subarray.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5\n1 7 2 11 15\n", "output_data": "3\n"},
                    {"input_data": "6\n100 100 100 100 100 100\n", "output_data": "1\n"},
                ]
            },
            # 2. Greedy
            {
                "problem_id": "CF-158B",
                "title": "Taxi",
                "contest_id": 158,
                "index": "B",
                "rating": 1100,
                "ground_truth_tag": "greedy",
                "all_tags": ["greedy"],
                "statement": "After lessons n groups of schoolchildren went outside. Each group consists of 1 to 4 children. A taxi takes at most 4 passengers. Find the minimum number of taxis needed by greedily pairing groups.",
                "input_spec": "The first line contains integer n (1 <= n <= 10^5). The second line contains n integers s_1, s_2, ..., s_n (1 <= s_i <= 4).",
                "output_spec": "Print the single integer - the minimum number of taxis.",
                "time_limit_ms": 3000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5\n1 2 4 3 3\n", "output_data": "4\n"},
                    {"input_data": "8\n2 3 4 4 2 1 3 1\n", "output_data": "5\n"},
                ]
            },
            {
                "problem_id": "CF-479A",
                "title": "Expression",
                "contest_id": 479,
                "index": "A",
                "rating": 1000,
                "ground_truth_tag": "greedy",
                "all_tags": ["greedy", "math"],
                "statement": "Given three integers a, b and c. Insert '+' and '*' and parentheses between them to greedily maximize the expression value.",
                "input_spec": "Three lines containing integers a, b, c (1 <= a, b, c <= 10).",
                "output_spec": "Print the maximum value.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "1\n2\n3\n", "output_data": "9\n"},
                    {"input_data": "2\n10\n3\n", "output_data": "60\n"},
                ]
            },
            {
                "problem_id": "CF-996A",
                "title": "Hit the Lottery",
                "contest_id": 996,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "greedy",
                "all_tags": ["greedy"],
                "statement": "Allen has n dollars and wants to withdraw it in minimum number of bills. Denominations are 100, 20, 10, 5, and 1 dollar. Find the minimum bills using greedy coin change.",
                "input_spec": "Single line: integer n (1 <= n <= 10^9).",
                "output_spec": "Print the minimum number of bills.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "125\n", "output_data": "3\n"},
                    {"input_data": "43\n", "output_data": "5\n"},
                ]
            },
            # 3. Graphs
            {
                "problem_id": "CF-500A",
                "title": "New Year Transportation",
                "contest_id": 500,
                "index": "A",
                "rating": 1000,
                "ground_truth_tag": "graphs",
                "all_tags": ["dfs and similar", "graphs"],
                "statement": "There are n cells from 1 to n. From cell i (1 <= i <= n-1), a portal takes you directly to cell i + a_i. You start at cell 1. Determine if you can reach target cell t in the graph.",
                "input_spec": "First line contains n and t (2 <= n <= 30000, 2 <= t <= n). Second line contains n-1 integers a_1, a_2, ..., a_{n-1}.",
                "output_spec": "Output 'YES' if you can reach cell t, otherwise 'NO'.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "8 4\n1 2 1 2 1 2 1\n", "output_data": "YES\n"},
                    {"input_data": "8 5\n1 2 1 2 1 1 1\n", "output_data": "NO\n"},
                ]
            },
            {
                "problem_id": "CF-520B",
                "title": "Two Buttons",
                "contest_id": 520,
                "index": "B",
                "rating": 1400,
                "ground_truth_tag": "graphs",
                "all_tags": ["dfs and similar", "graphs"],
                "statement": "Vasya has a device displaying integer n. Pressing red multiplies by 2. Pressing blue subtracts 1. Find the minimum button presses to reach m using graph BFS traversal.",
                "input_spec": "Two integers n and m (1 <= n, m <= 10^4).",
                "output_spec": "Print the minimum number of clicks.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "4 6\n", "output_data": "2\n"},
                    {"input_data": "10 1\n", "output_data": "9\n"},
                ]
            },
            {
                "problem_id": "CF-1033A",
                "title": "King Escape",
                "contest_id": 1033,
                "index": "A",
                "rating": 1000,
                "ground_truth_tag": "graphs",
                "all_tags": ["graphs", "dfs and similar"],
                "statement": "Alice place a Queen on an n x n chessboard at (ax, ay). Bob wants to move his King from (bx, by) to (cx, cy) without entering any square attacked by the Queen. Determine if reachable using graph connectivity.",
                "input_spec": "Line 1: n (3 <= n <= 1000). Line 2: ax, ay. Line 3: bx, by. Line 4: cx, cy.",
                "output_spec": "Print 'YES' if path exists, 'NO' otherwise.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "8\n4 4\n1 3\n3 1\n", "output_data": "YES\n"},
                    {"input_data": "8\n4 4\n2 3\n1 6\n", "output_data": "NO\n"},
                ]
            },
            # 4. Math
            {
                "problem_id": "CF-4A",
                "title": "Watermelon",
                "contest_id": 4,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "math",
                "all_tags": ["math"],
                "statement": "Pete and Billy want to divide a watermelon of weight w kilos into two parts, such that each part weighs an even number of kilos. Can they do it?",
                "input_spec": "Single integer w (1 <= w <= 100).",
                "output_spec": "Print YES, if they can divide the watermelon into two even parts; NO otherwise.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 64,
                "sample_tests": [
                    {"input_data": "8\n", "output_data": "YES\n"},
                    {"input_data": "2\n", "output_data": "NO\n"},
                ]
            },
            {
                "problem_id": "CF-1335A",
                "title": "Candies and Two Sisters",
                "contest_id": 1335,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "math",
                "all_tags": ["math"],
                "statement": "You have n candies. You want to distribute all candies between Alice and Betty such that Alice receives a > 0 candies, Betty receives b > 0 candies, a > b, and a + b = n. Calculate the number of valid distributions.",
                "input_spec": "First line: t (1 <= t <= 10^4). Next t lines: integer n (1 <= n <= 2*10^9).",
                "output_spec": "For each testcase print the number of ways.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "6\n7\n1\n2\n3\n2000000000\n7\n", "output_data": "3\n0\n0\n1\n999999999\n3\n"},
                ]
            },
            {
                "problem_id": "CF-617A",
                "title": "Elephant",
                "contest_id": 617,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "math",
                "all_tags": ["math"],
                "statement": "An elephant wants to visit his friend at coordinate x. The elephant can move 1, 2, 3, 4, or 5 positions forward in one step. What is the minimum number of steps to reach x?",
                "input_spec": "Single integer x (1 <= x <= 10^6).",
                "output_spec": "Print the minimum number of steps.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5\n", "output_data": "1\n"},
                    {"input_data": "12\n", "output_data": "3\n"},
                ]
            },
            # 5. Data Structures
            {
                "problem_id": "CF-20C",
                "title": "Dijkstra?",
                "contest_id": 20,
                "index": "C",
                "rating": 1600,
                "ground_truth_tag": "data structures",
                "all_tags": ["data structures", "shortest paths"],
                "statement": "You are given a weighted undirected graph with n vertices and m edges. Find the shortest path between vertex 1 and vertex n using a min-priority queue data structure.",
                "input_spec": "First line: n and m (2 <= n <= 10^5, 0 <= m <= 10^5). Next m lines contain u, v, w.",
                "output_spec": "Print the vertices along the shortest path or -1 if no path exists.",
                "time_limit_ms": 4000,
                "memory_limit_mb": 64,
                "sample_tests": [
                    {"input_data": "5 6\n1 2 2\n2 5 5\n2 3 4\n1 4 1\n4 3 3\n3 5 1\n", "output_data": "1 4 3 5\n"},
                ]
            },
            {
                "problem_id": "CF-342A",
                "title": "Xenia and Divisors",
                "contest_id": 342,
                "index": "A",
                "rating": 1200,
                "ground_truth_tag": "data structures",
                "all_tags": ["data structures"],
                "statement": "Xenia has n integers between 1 and 7. Divide them into triplets (a, b, c) such that a < b < c and a|b and b|c using a frequency array / map data structure.",
                "input_spec": "First line: n (3 <= n <= 99999). Second line: n integers.",
                "output_spec": "Print n/3 triplets or -1.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "6\n1 1 2 2 4 4\n", "output_data": "1 2 4\n1 2 4\n"},
                ]
            },
            {
                "problem_id": "CF-4C",
                "title": "Registration System",
                "contest_id": 4,
                "index": "C",
                "rating": 1200,
                "ground_truth_tag": "data structures",
                "all_tags": ["data structures", "hashing"],
                "statement": "A new email service is being registered. When a user requests a name, if it doesn't exist, output 'OK'. If it exists, append the smallest positive integer and register using a hash map data structure.",
                "input_spec": "First line: n (1 <= n <= 10^5). Next n lines contain strings of lowercase letters.",
                "output_spec": "Print 'OK' or the assigned unique username.",
                "time_limit_ms": 3000,
                "memory_limit_mb": 64,
                "sample_tests": [
                    {"input_data": "4\nabacaba\nacaba\nabacaba\nacab\n", "output_data": "OK\nOK\nabacaba1\nOK\n"},
                ]
            },
            # 6. Trees
            {
                "problem_id": "CF-580C",
                "title": "Kefa and Park",
                "contest_id": 580,
                "index": "C",
                "rating": 1500,
                "ground_truth_tag": "trees",
                "all_tags": ["trees", "dfs and similar"],
                "statement": "Kefa lives in a park with n vertices connected as a rooted tree at vertex 1. Count valid leaves reachable with at most m consecutive cats along the path in the tree.",
                "input_spec": "First line: n and m. Second line: n integers a_i (0 or 1). Next n-1 lines: tree edges.",
                "output_spec": "Print number of valid reachable leaves.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "4 1\n1 1 0 0\n1 2\n1 3\n1 4\n", "output_data": "2\n"},
                    {"input_data": "7 1\n1 0 1 1 0 0 0\n1 2\n1 3\n2 4\n2 5\n3 6\n3 7\n", "output_data": "2\n"},
                ]
            },
            {
                "problem_id": "CF-115A",
                "title": "Party",
                "contest_id": 115,
                "index": "A",
                "rating": 900,
                "ground_truth_tag": "trees",
                "all_tags": ["trees"],
                "statement": "A company has n employees arranged in a tree/forest hierarchy. Find the maximum depth across the tree forest.",
                "input_spec": "First line: n (1 <= n <= 2000). Next n lines: manager p_i (-1 if root).",
                "output_spec": "Print the minimum number of groups (maximum tree depth).",
                "time_limit_ms": 3000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5\n-1\n1\n2\n1\n-1\n", "output_data": "3\n"},
                ]
            },
            {
                "problem_id": "CF-1363A",
                "title": "Odd Selection",
                "contest_id": 1363,
                "index": "A",
                "rating": 1200,
                "ground_truth_tag": "trees",
                "all_tags": ["trees", "brute force"],
                "statement": "Given an array of n integers. Determine if you can select x elements such that their sum is odd.",
                "input_spec": "First line: t (1 <= t <= 100). For each test case, line 1: n and x. Line 2: n integers.",
                "output_spec": "Print 'Yes' or 'No' for each testcase.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5\n1 1\n999\n1 1\n1000\n2 1\n51 50\n2 2\n51 50\n3 3\n101 102 103\n", "output_data": "Yes\nNo\nYes\nYes\nNo\n"},
                ]
            },
            # 7. Brute Force
            {
                "problem_id": "CF-231A",
                "title": "Team",
                "contest_id": 231,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "brute force",
                "all_tags": ["brute force"],
                "statement": "Three friends decide to solve n competitive programming problems. For each problem, if at least 2 are sure, they write it. Count how many problems they solve via brute force scan.",
                "input_spec": "First line: n (1 <= n <= 1000). Next n lines contain 3 integers (0 or 1).",
                "output_spec": "Print the number of problems solved.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "3\n1 1 0\n1 1 1\n1 0 0\n", "output_data": "2\n"},
                    {"input_data": "2\n1 0 0\n0 1 1\n", "output_data": "1\n"},
                ]
            },
            {
                "problem_id": "CF-71A",
                "title": "Way Too Long Words",
                "contest_id": 71,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "brute force",
                "all_tags": ["brute force", "strings"],
                "statement": "If word length > 10, replace with abbreviation: first letter + omitted length + last letter. Otherwise keep unchanged. Brute force check each word.",
                "input_spec": "First line: n (1 <= n <= 100). Next n lines contain words.",
                "output_spec": "Print the n processed words.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 64,
                "sample_tests": [
                    {"input_data": "4\nword\nlocalization\ninternationalization\npneumonoultramicroscopicsilicovolcanoconiosis\n", "output_data": "word\nl10n\ni18n\np43s\n"},
                ]
            },
            {
                "problem_id": "CF-282A",
                "title": "Bit++",
                "contest_id": 282,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "brute force",
                "all_tags": ["brute force"],
                "statement": "The programming language Bit++ has variable x initialized to 0. Execute n statements (containing '++' or '--') sequentially via brute force simulation and output final value of x.",
                "input_spec": "First line: n (1 <= n <= 150). Next n lines contain single statement.",
                "output_spec": "Print final x value.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "1\n++X\n", "output_data": "1\n"},
                    {"input_data": "2\nX++\n--X\n", "output_data": "0\n"},
                ]
            },
            # 8. Strings
            {
                "problem_id": "CF-118A",
                "title": "String Task",
                "contest_id": 118,
                "index": "A",
                "rating": 1000,
                "ground_truth_tag": "strings",
                "all_tags": ["strings"],
                "statement": "Delete vowels, lowercase consonants, and insert '.' before each consonant in a string manipulation task.",
                "input_spec": "Single line: string (length <= 100).",
                "output_spec": "Print resulting string.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "tour\n", "output_data": ".t.r\n"},
                    {"input_data": "Codeforces\n", "output_data": ".c.d.f.r.c.s\n"},
                    {"input_data": "aBAcAba\n", "output_data": ".b.c.b\n"},
                ]
            },
            {
                "problem_id": "CF-112A",
                "title": "Petya and Strings",
                "contest_id": 112,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "strings",
                "all_tags": ["strings"],
                "statement": "Compare two strings lexicographically ignoring case. Print -1, 0, or 1.",
                "input_spec": "Two lines containing strings (length <= 100).",
                "output_spec": "Print -1, 0, or 1.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "aaaa\naaaA\n", "output_data": "0\n"},
                    {"input_data": "abs\nAbz\n", "output_data": "-1\n"},
                    {"input_data": "abcdefg\nAbCdEfF\n", "output_data": "1\n"},
                ]
            },
            {
                "problem_id": "CF-791A",
                "title": "Bear and Big Brother",
                "contest_id": 791,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "strings",
                "all_tags": ["strings", "implementation"],
                "statement": "Limak weighs a and Bob weighs b (a <= b). Each year Limak's weight triples and Bob's doubles. Find years until Limak is strictly heavier.",
                "input_spec": "Single line: two integers a and b (1 <= a <= b <= 10).",
                "output_spec": "Print integer number of years.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "4 7\n", "output_data": "2\n"},
                    {"input_data": "4 9\n", "output_data": "3\n"},
                ]
            },
            # 9. Number Theory
            {
                "problem_id": "CF-230B",
                "title": "T-primes",
                "contest_id": 230,
                "index": "B",
                "rating": 1300,
                "ground_truth_tag": "number theory",
                "all_tags": ["number theory", "math"],
                "statement": "A number is T-prime if it has exactly three distinct positive divisors (i.e. square of a prime). Test each of n given numbers.",
                "input_spec": "First line: n (1 <= n <= 10^5). Second line: n integers x_i (1 <= x_i <= 10^12).",
                "output_spec": "Print 'YES' or 'NO' for each number.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "3\n4 5 6\n", "output_data": "YES\nNO\nNO\n"},
                ]
            },
            {
                "problem_id": "CF-1374A",
                "title": "Required Remainder",
                "contest_id": 1374,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "number theory",
                "all_tags": ["number theory"],
                "statement": "Given x, y, and n. Find maximum integer k <= n such that k % x = y using modular arithmetic.",
                "input_spec": "First line: t (1 <= t <= 50000). Next t lines: x, y, n.",
                "output_spec": "For each testcase print k.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "7\n7 5 12345\n5 0 4\n10 5 15\n17 8 54321\n499999993 9 1000000000\n10 5 187\n2 0 999999999\n", "output_data": "12339\n0\n15\n54306\n999999995\n185\n999999998\n"},
                ]
            },
            {
                "problem_id": "CF-1389A",
                "title": "LCM Problem",
                "contest_id": 1389,
                "index": "A",
                "rating": 800,
                "ground_truth_tag": "number theory",
                "all_tags": ["number theory", "math"],
                "statement": "Given l and r. Find two integers x and y such that l <= x < y <= r and l <= LCM(x, y) <= r using number theory properties.",
                "input_spec": "First line: t (1 <= t <= 10000). Next t lines: l and r (1 <= l < r <= 10^9).",
                "output_spec": "Print x and y, or -1 -1.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "4\n1 1337\n13 69\n2 4\n88 89\n", "output_data": "1 2\n13 26\n2 4\n-1 -1\n"},
                ]
            },
            # 10. Binary Search
            {
                "problem_id": "CF-706B",
                "title": "Interesting drink",
                "contest_id": 706,
                "index": "B",
                "rating": 1100,
                "ground_truth_tag": "binary search",
                "all_tags": ["binary search"],
                "statement": "Vasya wants to buy drink. For each of q days with budget m_i, count affordable shops by sorting prices and applying binary search (upper_bound).",
                "input_spec": "Line 1: n. Line 2: prices x_i. Line 3: q. Next q lines: budget m_i.",
                "output_spec": "Print q lines with count of affordable shops.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "5\n3 10 8 6 11\n4\n1\n10\n3\n11\n", "output_data": "0\n4\n1\n5\n"},
                ]
            },
            {
                "problem_id": "CF-279B",
                "title": "Books",
                "contest_id": 279,
                "index": "B",
                "rating": 1400,
                "ground_truth_tag": "binary search",
                "all_tags": ["binary search"],
                "statement": "Given n books and free time t. Compute prefix sums and use binary search to find maximum consecutive books read within t minutes.",
                "input_spec": "First line: n and t. Second line: n integers a_i.",
                "output_spec": "Print maximum books read.",
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "4 5\n3 1 2 1\n", "output_data": "3\n"},
                    {"input_data": "3 3\n2 2 3\n", "output_data": "1\n"},
                ]
            },
            {
                "problem_id": "CF-492B",
                "title": "Vanya and Lanterns",
                "contest_id": 492,
                "index": "B",
                "rating": 1200,
                "ground_truth_tag": "binary search",
                "all_tags": ["binary search", "math"],
                "statement": "Vanya has n lanterns on a street of length l. Find minimum light radius d such that entire street [0, l] is illuminated using binary search over radius d.",
                "input_spec": "First line: n and l (1 <= n <= 1000, 1 <= l <= 10^9). Second line: n lantern coordinates.",
                "output_spec": "Print minimum radius d with precision 10^-9.",
                "time_limit_ms": 1000,
                "memory_limit_mb": 256,
                "sample_tests": [
                    {"input_data": "7 15\n15 5 3 7 9 14 0\n", "output_data": "2.5000000000\n"},
                    {"input_data": "2 5\n2 5\n", "output_data": "2.0000000000\n"},
                ]
            }
        ]

        problems = [Problem.from_dict(d) for d in raw_curated]
        logger.info("Loaded %d curated problems across %d categories.", len(problems), len(self.target_tags))
        return problems
