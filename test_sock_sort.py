import unittest

import cv2
import numpy as np

from sock_sort import detect, plan, print_difference, validate_layout


class SockSortTest(unittest.TestCase):
    boxes = [[400, 5+100*i, 90, 90] for i in range(4)]
    left = [0, 0, 90, 400]

    def make_socks(self, count, different=False, visible=True):
        empty = np.full((420, 500, 3), 230, np.uint8)
        frame = empty.copy()
        for index in range(count):
            x = 110 + (index % 6)*46
            y = 30 + (index // 6)*100
            if not visible and index == 0:
                y = 0
            cv2.rectangle(frame, (x, y), (x+31, y+64), (40, 60, 130), -1)
            char = 'B' if different and index == count-1 else 'A'
            cv2.putText(frame, char, (x+3, y+44), cv2.FONT_HERSHEY_SIMPLEX, 1, (245,)*3, 2)
        return detect(frame, empty, (100, 0, 290, 420))

    def test_print_comparison_and_two_or_three_matching_socks(self):
        socks = self.make_socks(3, different=True)
        self.assertEqual(len(socks), 3)
        self.assertEqual(print_difference(socks[0], socks[1]), 0)
        self.assertGreater(print_difference(socks[0], socks[2]), 0)
        placements, held = plan(socks, self.boxes, self.left,
                                lambda pair, _: 'm' if pair == (1, 2) else 'd')
        self.assertEqual([(p['socks'], p['destination']) for p in placements],
                         [([1, 2], 'box_1'), ([3], 'left')])
        self.assertFalse(held)
        trio, _ = plan(self.make_socks(3), self.boxes, self.left, lambda *_: 'm')
        self.assertEqual(trio[0]['socks'], [1, 2, 3])

    def test_ambiguous_and_cropped_socks_are_held(self):
        socks = self.make_socks(3)
        placements, held = plan(socks, self.boxes, self.left,
                                lambda pair, _: 'm' if pair != (1, 3) else 'u')
        self.assertFalse(placements)
        self.assertEqual({i for group in held for i in group['socks']}, {1, 2, 3})
        placements, held = plan(self.make_socks(1, visible=False), self.boxes, self.left,
                                lambda *_: 'd')
        self.assertFalse(placements)
        self.assertEqual(held[0]['reason'], 'reinspect')

    def test_box_limit_and_layout(self):
        placements, held = plan(self.make_socks(10), self.boxes, self.left,
                                lambda pair, _: 'm' if pair[0] % 2 and pair[1] == pair[0]+1 else 'd')
        self.assertEqual(len(placements), 4)
        self.assertEqual(held[0]['reason'], 'no empty right box')
        with self.assertRaises(ValueError):
            validate_layout({'pickup': [100, 0, 290, 420], 'left': self.left,
                             'boxes': self.boxes[:3]}, (420, 500, 3))


if __name__ == '__main__':
    unittest.main()
