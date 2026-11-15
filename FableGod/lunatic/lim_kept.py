# IDEAS_1115_limits: kept per 32-row check from per-band accuracies (THINK_1115 table), product rule:
# kept = bonus + sum_d prod_{i<=d} a_i over depths 0..30 (bands 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30).
import numpy as np
W = [1, 2, 4, 4, 4, 8, 8]
def kept(bands, bonus=1.0):
    a = np.repeat(np.array(bands) / 100, W)
    return bonus + np.cumprod(a).sum()
P = {
 't30b chain [K]':                         [92.0, 91.5, 92.4, 94.3, 96.6, 96.7, 97.7],
 'true state [K]':                          [93.2, 95.3, 96.4, 97.8, 98.8, 99.2, 99.3],
 'flat 97.8':                               [97.8] * 7,
 'flat 98.0':                               [98.0] * 7,
 'true state, bands 0-6 lifted to 97.8':    [97.8, 97.8, 97.8, 97.8, 98.8, 99.2, 99.3],
 'tree-16 bands 0-2 + true state beyond':   [98.6, 96.4, 96.4, 97.8, 98.8, 99.2, 99.3],
 'true state, band 0 only lifted to 98.6':  [98.6, 95.3, 96.4, 97.8, 98.8, 99.2, 99.3],
 'true state, bands 0-6 perfect (100)':     [100, 100, 100, 97.8, 98.8, 99.2, 99.3],
 'chain, bands 7-30 at true state':         [92.0, 91.5, 92.4, 97.8, 98.8, 99.2, 99.3],
 'true state, bands 7-30 at chain':         [93.2, 95.3, 96.4, 94.3, 96.6, 96.7, 97.7],
}
for k, v in P.items():
    kp = kept(v)
    print(f"{k:42s} kept/32 {kp:5.2f} | 52 ms: {kp / 52.0 * 1e3:4.0f} tok/s | 59.5 ms (floor + 7.5 overhead): {kp / 59.5 * 1e3:4.0f} | 25 ms: {kp / 25 * 1e3:5.0f}")
print('\n-- lossy close-call acceptance at bands 0-6 (near-perfect there), deep bands as named; x0.956 = calibration of the')
print('   product rule to the measured kept (13.77 / 14.44, 20.77 / 21.67)')
for k, v in {'bands 0-6 at 99.5, deep = t30b chain':      [99.5, 99.5, 99.5, 94.3, 96.6, 96.7, 97.7],
             'bands 0-6 at 99.5, deep = chain + audit':   [99.5, 99.5, 99.5, 94.3, 96.6, 98.3, 98.3],
             'bands 0-6 at 99.5, deep = true state':      [99.5, 99.5, 99.5, 97.8, 98.8, 99.2, 99.3],
             'all bands 99.0 (close calls everywhere)':   [99.0] * 7}.items():
    kp = kept(v)
    print(f"{k:42s} kept/32 {kp:5.2f} (cal. {kp * 0.956:5.2f}) | 25 ms: {kp * 0.956 / 25 * 1e3:5.0f} tok/s | 20 ms: {kp * 0.956 / 20 * 1e3:5.0f}")
