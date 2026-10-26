import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, time, timezone

from reliability import save_state, load_state
from service_worker import polling_stale, run_child, check_deadline, should_check


class SupervisorTests(unittest.TestCase):
    def test_stale_or_wrong_process_heartbeat_and_startup_grace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'health.json'
            self.assertFalse(polling_stale(path, 42, 100, 300, now=399))
            self.assertTrue(polling_stale(path, 42, 100, 300, now=401))
            save_state(path, {'pid': 42, 'polled_at': 400})
            self.assertFalse(polling_stale(path, 42, 100, 300, now=401))
            save_state(path, {'pid': 99, 'polled_at': 400})
            self.assertTrue(polling_stale(path, 42, 100, 300, now=401))

    def test_hung_child_is_terminated_and_nonzero_exit_is_preserved(self):
        self.assertEqual(run_child([sys.executable, '-c', 'import time; time.sleep(60)'], .05), 124)
        self.assertEqual(run_child([sys.executable, '-c', 'raise SystemExit(7)'], 10), 7)

    def test_daily_alert_sent_once_and_only_after_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            state, alert = Path(folder) / 'state.json', Path(folder) / 'alert.json'
            sent = []
            check_deadline(state, alert, datetime(2026, 10, 11, 17, tzinfo=timezone.utc), time(18), sent.append)
            self.assertFalse(sent)
            now = datetime(2026, 10, 11, 18, tzinfo=timezone.utc)
            check_deadline(state, alert, now, time(18), sent.append)
            check_deadline(state, alert, now, time(18), sent.append)
            self.assertEqual(len(sent), 1)
            save_state(state, {'last_delivery_date': '2026-10-12'})
            check_deadline(state, alert, datetime(2026, 10, 12, 18, tzinfo=timezone.utc), time(18), sent.append)
            self.assertEqual(len(sent), 1)

    def test_no_alert_when_last_nights_post_was_delivered_today(self):
        with tempfile.TemporaryDirectory() as folder:
            state, alert = Path(folder) / 'state.json', Path(folder) / 'alert.json'
            sent = []
            delivered_at = datetime(2026, 10, 11, 8, 15, tzinfo=timezone.utc).timestamp()
            save_state(state, {'last_delivery_date': '2026-10-10', 'last_success_at': delivered_at})
            check_deadline(state, alert, datetime(2026, 10, 11, 18, tzinfo=timezone.utc), time(18), sent.append)
            self.assertFalse(sent)
            # The next day nothing has been delivered yet.
            check_deadline(state, alert, datetime(2026, 10, 12, 18, tzinfo=timezone.utc), time(18), sent.append)
            self.assertEqual(len(sent), 1)

    def test_notification_failure_remains_retryable(self):
        with tempfile.TemporaryDirectory() as folder:
            state, alert = Path(folder) / 'state.json', Path(folder) / 'alert.json'
            def fail(text):
                raise OSError('offline')
            with self.assertRaises(OSError):
                check_deadline(state, alert, datetime(2026, 10, 11, 18), time(18), fail)
            self.assertEqual(load_state(alert), {})

    def test_checks_stop_at_six_and_after_success_and_resume_next_day(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / 'state.json'
            def eligible(day, hour, minute=0):
                return should_check(state, datetime(2026, 10, day, hour, minute), time(8), time(18))
            self.assertFalse(eligible(11, 7, 59))
            self.assertTrue(eligible(11, 8))
            self.assertTrue(eligible(11, 17, 59))
            self.assertFalse(eligible(11, 18))
            self.assertFalse(eligible(11, 23))
            save_state(state, {'last_delivery_date': '2026-10-10'})
            self.assertTrue(eligible(11, 9))  # Yesterday's backlog does not satisfy today.
            save_state(state, {'last_delivery_date': '2026-10-11'})
            self.assertFalse(eligible(11, 9))  # Also holds after a worker restart.
            self.assertTrue(eligible(12, 8))
