"""Complete Hamming-radius lookup using k+1 disjoint bands.

At distance <= k at least one band is unchanged. All candidates are verified by
Hamming distance; dense buckets can still require linear work.
"""
class SimHashIndex:
    def __init__(self, radius=3, bits=64):
        if type(bits) is not int or type(radius) is not int or not 0 <= radius < bits:
            raise ValueError("require integer 0 <= radius < bits")
        self.radius, self.bits = radius, bits
        self.bands, self.buckets = [], {}
        offset = 0
        for i in range(radius + 1):
            width = bits // (radius + 1) + (i < bits % (radius + 1))
            self.bands.append((offset, (1 << width) - 1))
            offset += width

    def candidates(self, value):
        if type(value) is not int or not 0 <= value < 1 << self.bits:
            raise ValueError("fingerprint outside bit width")
        result = set()
        for i, (offset, mask) in enumerate(self.bands):
            result.update(self.buckets.get((i, (value >> offset) & mask), ()))
        return result

    def contains_near(self, value):
        return any((value ^ other).bit_count() <= self.radius for other in self.candidates(value))

    def add(self, value):
        self.candidates(value)  # validate before changing state
        for i, (offset, mask) in enumerate(self.bands):
            self.buckets.setdefault((i, (value >> offset) & mask), set()).add(value)
