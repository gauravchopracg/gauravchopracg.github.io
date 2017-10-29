---
layout: post
title: "Parallel Scan for Exclusive Prefix Sum"
---

[Prefix Sum](https://en.wikipedia.org/wiki/Prefix_sum) problem is to compute the sum of all the previous elements in an array. Specifically, _exclusive prefix sum_ would compute all the strictly previous (self-exclusive) elements. For example,

| Input  | 1 | 2 | 3 | 4 | 5  |
|--------|---|---|---|---|----|
| Output | 0 | 1 | 3 | 6 | 10 |

## Parallel Scan

The parallel algorithm to solve this is called _parallel scan_. It can be generalized to more commutative operators with an identity, such as multiplication, and, etc. It consists of two phases: _upsweep_ and _downsweep_.

_upsweep_ is to add the element to the other element that is $2^p$ away from it. For simplicity, the edge check is omitted.

```c
void upsweep(int* a, int i, int p) {
  int d = 1 << p;
  a[i+d] += a[i];
}
```

_downsweep_ is similar to _upsweep_, except that it will swap the element that was $2^p$ away to the current element.

```c
void downsweep(int* a, int i, int p) {
  int d = 1 << p;
  int tmp = a[i];
  a[i] = a[i+d];
  a[i+d] += tmp;
}
```

For simplicity, let's assume the length of array is magnitude of 2. If not, we can always allocate a larger array, with the rest of it wasted. Here's the pseudo code:

```c
// Assume |a| has length 1 << q.
void prefix_sum(int* a, int q) {
  int l = 1 << q;
  
  // Upsweep phase
  for (int p = 0; p < q; p++) {
    int d = 1 << p;
    int dd = d << 1;
    parallel_for (int i = d - 1; i < l - d; i += dd) {
      upsweep(a, i, p);
    }
  }
  
  // Reset the last element
  a[l-1] = 0;
  
  // Downsweep phase
  for (int p = q - 1; p >= 0; p--) {
    int d = 1 << p;
    int dd = d << 1;
    parallel_for (int i = d - 1; i < l - 1; i += dd) {
      downsweep(a, i, p);
    }
  }
}
```

## Example

The following is an example of exclusive sum from 0 to 7. The first column is `d` and `dd`, where `d` is the distance within a sweep, and `dd` is the distance for the next sweep. The elements that are filled are those being updated.


|        | 0 | 1 | 2 | 3 | 4 | 5  | 6  | 7  |
|--------|---|---|---|---|---|----|----|----|
| (1, 2) |   | 1 |   | 5 |   | 9  |    | 13 |
| (2, 4) |   |   |   | 6 |   |    |    | 22 |
| (4, 8) |   |   |   |   |   |    |    | 28 |
|        | 0 | 1 | 2 | 6 | 4 | 9  | 6  | 0  |
| (4, 8) |   |   |   | 0 |   |    |    | 6  |
| (2, 4) |   | 0 |   | 1 |   | 6  |    | 15 |
| (1, 2) | 0 | 0 | 1 | 3 | 6 | 10 | 15 | 21 |


## Analysis

See [wiki](https://en.wikipedia.org/wiki/Analysis_of_parallel_algorithms) for definition of _span_ and _work_. The _span_ of the algorithm is $O(\log N)$. In other words, given infinite number of processors, the algorithm can finish in $O(\log N)$ steps. The _work_ is $O(N \log N)$. In other words, given only one processor, the algorithm needs $O(N \log N)$ steps.

## Reference

* Kayvon's [note](http://15418.courses.cs.cmu.edu/spring2016/lecture/exclusivescan).
* 15418 [handout](http://15418.courses.cs.cmu.edu/spring2016/article/4); search for _Exclusive Prefix Sum_.
* My [homework](https://github.com/misaka-10032/PP/blob/master/assignment2/scan/scan.cu).