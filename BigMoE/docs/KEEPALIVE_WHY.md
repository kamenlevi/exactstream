# For MAIN 8.1 (v2): why the SSD keepalive works, what is measured, what is cheap to measure next
(Mon 9 Nov 17:4x; for the editor, due Tue; sources = BM:data/gate/gate_facts.md section 9, which cites every raw file)

**Why it works: what is measured and what is not.** On this machine's internal SSD, measured in micro-benchmarks that
load no model: after 0.10 ms of idle there is no penalty, from 0.25 ms on the full one [K, MS:data/ssdmicro.txt S4]; the
penalty grows with the gap, +50-65 us after 0.25-1 ms and +100-120 us after 20 ms [K, T93]; a CPU core kept busy without
I/O changes nothing [K, T93]; 4 KB reads at a real spacing of at most 340 us hold the penalty to 10 us or less, and larger
reads hold the drive awake longer (a 16 KB read at 412 us leaves +8.6 us, a 64 KB read at 450 us +4.6 us)
[K, MS:data/ssdmicro2.txt K]; a single 4 KB read must finish before the burst (100 us ahead removes 22 %, 200 us ahead
97.5 %, 400 us ahead 78 %) [K, T47]; and reads that end 0.25 ms or more before the next demand read let the drive doze
again [K, MS:data/ssdmicro3.txt]. Together these put the effect in the I/O path below the CPU and fit a low-power state
that is entered after about 0.10-0.25 ms of idle and costs about 45-65 us to leave [G]. NOT tested: which part idles (the
NVMe controller's power state, the PCIe link, the SoC's storage controller, or the kernel's I/O path); the exact
threshold; whether a deeper state follows after 20 ms; the energy cost of about 5,600 reads per second at the 50 us
setting [K, MS:data/ssdmicro.txt S5]; and whether the power source, Low Power Mode, temperature, the macOS version or
another Mac change any of it. The engine's own gain is known only inside the +12.1 % bundle: no isolated A/B of the
keepalive exists on the final engine.

**Cheap to measure next (no model; the existing ssdmicro harness; minutes each):**
1. A 10 us-step idle-gap sweep from 0.10 to 0.30 ms, with back-to-back controls interleaved: where the threshold is, and
   whether the penalty is a step or a ramp.
2. Gaps of 50 ms, 0.1 s, 1 s and 10 s: is there a second, deeper state?
3. The keepalive as a page-cache HIT on the same 4 KB block (a clean control: it should NOT keep the drive awake if the
   effect is in the device path) against 512 B / 4 KB / 16 KB device reads, same and far blocks.
4. Deadline-timed spacing (spin to a target, not usleep) from 150 to 400 us in 25 us steps: the exact hold-off curve.
5. One read against 8- and 16-read bursts after the same gap: is the wake paid once per burst?
6. Energy: keepalive on vs off at the engine's gap pattern (powermetrics needs root; otherwise battery drain over a fixed
   duty cycle).
7. AC vs battery and Low Power Mode, one run each.
8. A GPU bandwidth kernel with the keepalive on vs off (~22 MB/s of reads: expected negligible, never measured).
9. The wake-ahead ping vs the trickle at the engine's measured gap distribution (median 0.73 ms), counting reads per
   check (~46 vs ~300).
10. The same harness on the 36 GB M5 Max and on an external NVMe drive: is the penalty specific to Apple's internal SSD?
