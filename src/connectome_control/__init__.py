"""connectome-control: C. elegans connectome cart-pole, protocol v4.

Protocol v4 constants are code, not convention. Change them only with a
protocol version bump.
"""

PROTOCOL_VERSION = "v4"

# --- delay canon: action delay in MILLISECONDS of wall-clock time,
# converted to control steps at the loop rate. The command computed at
# tick t is applied DELAY_STEPS ticks later (zero-order hold).
DELAY_MS = 20.0
HZ = 50.0
DT = 1.0 / HZ
DELAY_STEPS = round(DELAY_MS / 1000.0 * HZ)
assert DELAY_STEPS == 1, "generalise the pending-action buffer before other rates"

F_MAX = 18.0          # actuator saturation (N)
TRACK = 3.0           # track half-length (m)
UP_TOL_DEG = 12.0     # upright band
HOLD_S = 3.0          # "held" = upright for the final HOLD_S seconds

# --- v4 amendments (verified in the prototype):
# * positions-only observations; NO pending-action inputs for imitation-
#   trained policies (causal confusion: 4% held with, 97% without).
# * teachers are delay-aware: predict one exact RK4 step under the pending
#   action, then apply the undelayed law.
# * catchable-approach threshold (from the v4 teacher's first-pass catch
#   rate at band entry): 95% at 0.5 rad/s, 90% at 1.0.
CATCHABLE_THR = 1.0   # rad/s at band entry (90% teacher reference)
CATCHABLE_THR_STRICT = 0.5
