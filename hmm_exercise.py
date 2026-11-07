import itertools
import json
import math

from Bio import SeqIO

# TASK 1


def log2(p):
    return -math.inf if p == 0 else math.log2(p)


class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e

        # Check 1: pi, a and e, same the keys of states
        check1 = (set(pi) == set(states)
                  and set(a) == set(states)
                  and set(e) == set(states))

        # Check 2: every row of a, and every row of e
        check2 = True
        for k in states:
            if set(a[k]) != set(states):
                check2 = False
            if set(e[k]) != set(alphabet):
                check2 = False

        # Check 3: pi + every row of a + e = 1
        check3 = math.isclose(sum(pi.values()), 1)
        for k in states:
            if not math.isclose(sum(a[k].values()), 1):
                check3 = False
            if not math.isclose(sum(e[k].values()), 1):
                check3 = False

        # printing the checks!!
        n = 1
        for check in [check1, check2, check3]:
            if check:
                print("check", n, "ok")
            else:
                print("check", n, "not ok")
            n += 1

        self.log_pi = {k: log2(pi[k]) for k in states}
        self.log_a = {k: {l: log2(a[k][l]) for l in states} for k in states}
        self.log_e = {k: {c: log2(e[k][c]) for c in alphabet} for k in states}

    def joint_probability(self, x, z):
        p = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            p = p * self.a[z[i - 1]][z[i]] * self.e[z[i]][x[i]]
        return p

    def log_joint(self, x, z):
        s = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            s = s + self.log_a[z[i - 1]][z[i]] + self.log_e[z[i]][x[i]]
        return s

    def brute_force(self, x):
        best_path = None
        best_p = -1.0
        total = 0.0
        for tup in itertools.product(self.states, repeat=len(x)):
            z = "".join(tup)
            p = self.joint_probability(x, z)
            total = total + p
            if p > best_p:
                best_p = p
                best_path = z
        return best_path, best_p, total

    # TASK 2

    def viterbi(self, x):
        v = []
        back = []

        v0 = {}
        b0 = {}
        for k in self.states:
            v0[k] = self.log_pi[k] + self.log_e[k][x[0]]
            b0[k] = None
        v.append(v0)
        back.append(b0)

        for i in range(1, len(x)):
            vi = {}
            bi = {}
            for k in self.states:
                best_score = -math.inf
                best_prev = self.states[0]
                for l in self.states:
                    score = v[i - 1][l] + self.log_a[l][k]
                    if score > best_score:
                        best_score = score
                        best_prev = l
                vi[k] = best_score + self.log_e[k][x[i]]
                bi[k] = best_prev
            v.append(vi)
            back.append(bi)

        last = len(x) - 1
        best_score = -math.inf
        best_state = self.states[0]
        for k in self.states:
            if v[last][k] > best_score:
                best_score = v[last][k]
                best_state = k

        path = [best_state]
        for i in range(last, 0, -1):
            best_state = back[i][best_state]
            path.append(best_state)
        path.reverse()

        return "".join(path), best_score

    # TASK 3

    def forward(self, x):
        f = []

        f0 = {}
        for k in self.states:
            f0[k] = self.log_pi[k] + self.log_e[k][x[0]]
        f.append(f0)

        for i in range(1, len(x)):
            fi = {}
            for k in self.states:
                s = -math.inf
                for l in self.states:
                    s = log_sum(s, f[i - 1][l] + self.log_a[l][k])
                fi[k] = s + self.log_e[k][x[i]]
            f.append(fi)

        total = -math.inf
        for k in self.states:
            total = log_sum(total, f[len(x) - 1][k])

        return f, total

    def backward(self, x):
        n = len(x)
        b = [None] * n

        b[n - 1] = {}
        for k in self.states:
            b[n - 1][k] = 0.0

        for i in range(n - 2, -1, -1):
            bi = {}
            for k in self.states:
                s = -math.inf
                for l in self.states:
                    s = log_sum(s, self.log_a[k][l]
                                + self.log_e[l][x[i + 1]]
                                + b[i + 1][l])
                bi[k] = s
            b[i] = bi

        total = -math.inf
        for k in self.states:
            total = log_sum(total, self.log_pi[k]
                            + self.log_e[k][x[0]]
                            + b[0][k])

        return b, total

    def posterior(self, x):
        f, total = self.forward(x)
        post = []
        b, total_b = self.backward(x)
        for i in range(len(x)):
            col = {}
            for k in self.states:
                col[k] = 2 ** (f[i][k] + b[i][k] - total)
            post.append(col)
        return post

    def posterior_path(self, x):
        post = self.posterior(x)
        path = ""
        for i in range(len(x)):
            best_p = -1.0
            best_state = self.states[0]
            for k in self.states:
                if post[i][k] > best_p:
                    best_p = post[i][k]
                    best_state = k
            path = path + best_state
        return path


def read_model(path):
    with open(path) as f:
        m = json.load(f)
    return HMM(m["states"], m["alphabet"], m["pi"], m["a"], m["e"])


def log_sum(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    big = max(a, b)
    small = min(a, b)
    return big + math.log2(1 + 2 ** (small - big))


def agreement(z1, z2):
    same = 0
    for i in range(len(z1)):
        if z1[i] == z2[i]:
            same = same + 1
    return same / len(z1)


# DECODE

toy = read_model("toy_model.json")

seqs = {}
for record in SeqIO.parse("sequences.fasta", "fasta"):
    seqs[record.id] = str(record.seq)

true = {}
for record in SeqIO.parse("true_paths.fasta", "fasta"):
    true[record.id] = str(record.seq)

for id in seqs:
    x = seqs[id]
    vit_path, vit_score = toy.viterbi(x)
    post_path = toy.posterior_path(x)
    print(id, len(x), agreement(vit_path, true[id]), agreement(post_path, true[id]))

