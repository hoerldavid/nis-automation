"""
Small retry helper: run a callable with backoff, retrying on selected
exceptions.

Used by the auto-FRAP pipeline around NIS macro calls, where a
transient failure (timeout, aborted macro read-back, OS hiccup) is
worth a retry but a persistent one should surface as-is. The helper is
exception-driven only: it never inspects or validates the result, which
is passed through unchanged.
"""
import time


def run_with_retries(fn, what, retry_on=Exception, delays=(0, 2, 4)):
    """
    Call fn() up to len(delays) times, retrying on matching exceptions.

    Parameters
    ----------
    fn: callable
        zero-arg callable holding the retried operation
    what: str
        short tag for the log lines, e.g. 'move_stage'
    retry_on: Exception type or tuple of types
        exception types that trigger a retry; any other exception
        propagates immediately (no retry)
    delays: sequence of float
        delay in seconds before each attempt (delays[0] before the
        first); the number of attempts is len(delays)

    Returns
    -------
    the result of the first successful fn() call, passed through
    unchanged

    Raises
    ------
    the first non-matching exception (immediately) or the last
    exception matching retry_on (after the final attempt)
    """
    if not delays:
        return fn()
    last_exc = None
    for attempt, delay in enumerate(delays, start=1):
        try:
            if delay:
                time.sleep(delay)
            result = fn()
            if attempt > 1:
                print(f'[{what}] succeeded on attempt {attempt}', flush=True)
            return result
        except retry_on as e:
            last_exc = e
            if attempt == len(delays):
                break
            print(f'[{what}] attempt {attempt} failed: {e!r}, '
                  f'retry in {delays[attempt]}s', flush=True)
    raise last_exc
