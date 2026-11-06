"""Kiểm tra crawler 3 file bằng HTML và driver giả; không truy cập Facebook."""
import csv
from pathlib import Path
import sys
import threading

import pytest
from selenium.common.exceptions import TimeoutException

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phien_ban_moi" / "crawl data"))
import crawl_comments as crawler
import get_post_urls as auth

URL = "https://www.facebook.com/PhimMuaDo/posts/123/"


def test_extract_short_nested_and_image_emoji():
    html = '''<div role="article"><div dir="auto">Nội dung bài viết</div>
      <div data-commentid="1"><a><span dir="auto">Tên tác giả</span></a>
        <div data-ad-rendering-role="comment_message">Dở</div>
        <div data-comment-id="2"><div data-ad-rendering-role="comment_message">
          <img alt="😀"></div></div>
      </div></div>'''
    rows = crawler.extract_comments(html, URL)
    assert [(row["comment_id"], row["comment"]) for row in rows] == [("1", "Dở"), ("2", "😀")]
    assert rows[1]["parent_comment_id"] == "1"


def test_extract_permalink_identity_and_exclude_author():
    html = '''<div role="article" aria-label="Bình luận của Một người">
        <a href="/profile.php?id=1"><span dir="auto">Tên tác giả</span></a>
        <div dir="auto">Hay</div>
        <a href="?comment_id=12"><span dir="auto">1 giờ</span></a>
        <div role="button"><span dir="auto">Thích</span></div>
      </div>'''
    rows = crawler.extract_comments(html, URL)
    assert len(rows) == 1
    assert rows[0]["comment_id"] == "12"
    assert rows[0]["comment"] == "Hay"


def test_login_modal_on_unchanged_url():
    class Driver:
        current_url = URL
        def execute_script(self, script):
            return True
    with pytest.raises(RuntimeError, match="hộp đăng nhập"):
        auth.ensure_access(Driver())


def test_scroll_saves_each_change_and_stops_on_repeated_buttons(monkeypatch):
    class Stop:
        def is_set(self): return False
        def wait(self, seconds): return False
    class Driver:
        current_url = URL
        turn = 0
        def get(self, url): pass
        @property
        def page_source(self):
            return f'<div data-commentid="{min(self.turn, 45)}"><div data-ad-rendering-role="comment_message">Hay</div></div>'
        def execute_script(self, script):
            if script != crawler.SCROLL_COMMENTS_SCRIPT:
                return False
            self.turn += 1
            return {"at_bottom": self.turn >= 45, "moved": self.turn < 45}
    monkeypatch.setattr(crawler, "select_all_comments", lambda driver, stop=None: True)
    monkeypatch.setattr(crawler, "wait_until", lambda stop, condition, **kwargs: condition() or None)
    # Nút vẫn bấm được nhưng không tạo thêm dữ liệu: crawler phải dừng.
    monkeypatch.setattr(crawler, "expand_comments", lambda driver: 1)
    driver, rows, snapshots = Driver(), {}, []
    crawler.crawl_post(driver, URL, None, Stop(), rows,
                       on_progress=lambda data: snapshots.append(len(data)))
    assert len(rows) == 46
    assert snapshots[-1] == 46
    assert driver.turn == 48


def test_five_workers_single_writer_partial_results(monkeypatch, tmp_path, capsys):
    barrier = threading.Barrier(5)
    owners, closed, writers = [], [], []
    urls = [URL.replace("123", str(i)) for i in range(5)]
    def make_driver(browser):
        owner = threading.get_ident()
        owners.append(owner)
        return owner
    def close_driver(owner):
        assert threading.get_ident() == owner
        closed.append(owner)
    def login(owner, cookies):
        assert threading.get_ident() == owner
    def crawl_post(owner, url, max_scrolls, stop, collected, on_progress=None):
        assert threading.get_ident() == owner
        barrier.wait(timeout=10)
        row = {"comment": "Hay", "comment_id": "same-id", "post_url": url,
               "parent_comment_id": "", "collected_at": "now"}
        collected[(url, "same-id")] = row
        on_progress([row])
        if url == urls[0]:
            raise TimeoutException("mock")
    original_save = crawler.save_rows
    def save(path, rows):
        writers.append(threading.get_ident())
        original_save(path, rows)
    monkeypatch.setattr(crawler, "make_driver", make_driver)
    monkeypatch.setattr(crawler, "close_driver", close_driver)
    monkeypatch.setattr(crawler, "login", login)
    monkeypatch.setattr(crawler, "crawl_post", crawl_post)
    monkeypatch.setattr(crawler, "save_rows", save)
    output = tmp_path / "comments.csv"
    rows = {}
    crawler.crawl_parallel(urls, [], "edge", 5, None, output, rows)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[0] == "comment"
        saved = list(reader)
    assert len(saved) == 5
    assert len(set(owners)) == 5
    assert sorted(owners) == sorted(closed)
    assert set(writers) == {threading.get_ident()}
    log = capsys.readouterr().out
    assert log.count("XONG POST:") == 4
    assert log.count("POST LỖI/DỪNG:") == 1
    assert f"XONG POST: {urls[0]}\n" not in log


def test_existing_csv_old_column_order_still_loads(monkeypatch, tmp_path):
    source = tmp_path / "post_urls.txt"
    source.write_text(URL + "\n", encoding="utf-8")
    output = tmp_path / "comments.csv"
    old_columns = ["comment_id", "post_url", "comment", "parent_comment_id", "collected_at"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow(dict(zip(old_columns, ["12", URL, "Hay", "", "now"])))
    monkeypatch.setattr(sys, "argv", ["crawl_comments.py", "--input", str(source), "--output", str(output)])
    monkeypatch.setattr(crawler, "read_cookies", lambda path: [])
    def run(urls, cookies, browser, workers, max_scrolls, path, rows):
        assert rows[(URL, "12")]["comment"] == "Hay"
        crawler.save_rows(path, rows)
    monkeypatch.setattr(crawler, "crawl_parallel", run)
    crawler.main()
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[0] == "comment"
        assert list(reader)[0]["comment"] == "Hay"


@pytest.mark.parametrize("changes_filter", [True, False])
def test_select_all_verifies_filter_change(monkeypatch, changes_filter):
    class Button:
        def __init__(self, callback): self.callback = callback
        def click(self): self.callback()
        def send_keys(self, keys): pass
    class Driver:
        current_url = URL
        label = "phù hợp nhất"
        opened = False
        def __init__(self):
            self.switch_to = self
            self.active_element = Button(lambda: None)
        def open(self): self.opened = True
        def choose(self):
            self.opened = False
            if changes_filter: self.label = "tất cả bình luận"
        def execute_script(self, script):
            if script == crawler.FILTER_CONTROL_SCRIPT:
                return {"label": self.label, "element": Button(self.open)}
            if script == crawler.ALL_COMMENTS_OPTION_SCRIPT:
                assert self.opened
                return Button(self.choose)
            return False
    monkeypatch.setattr(crawler.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(crawler, "wait_until", lambda stop, condition, **kwargs: condition() or None)
    assert crawler.select_all_comments(Driver()) is changes_filter


def test_reel_opens_drawer_before_extracting(monkeypatch):
    class Stop:
        def is_set(self): return False
        def wait(self, seconds): return False
    class Button:
        def __init__(self, driver): self.driver = driver
        def click(self): self.driver.opened = True
    class Driver:
        current_url = "https://www.facebook.com/reel/123/"
        opened = False
        def get(self, url): pass
        @property
        def page_source(self):
            if not self.opened: return '<video></video>'
            # Reel không bắt buộc có role=article.
            return '<section aria-label="Bình luận của A"><div data-ad-rendering-role="comment_message">Hay</div></section>'
        def execute_script(self, script, *args):
            if script == crawler.COMMENTS_READY_SCRIPT: return self.opened
            if script == crawler.OPEN_COMMENTS_SCRIPT: return Button(self)
            return False
    monkeypatch.setattr(crawler, "select_all_comments", lambda driver, stop=None: True)
    monkeypatch.setattr(crawler, "wait_until", lambda stop, condition, **kwargs: condition() or None)
    rows, driver = {}, Driver()
    crawler.crawl_post(driver, driver.current_url, 0, Stop(), rows)
    assert driver.opened
    assert len(rows) == 1
    assert next(iter(rows.values()))["comment"] == "Hay"


def test_reel_unopened_is_error_not_zero_success(monkeypatch):
    class Stop:
        def is_set(self): return False
        def wait(self, seconds): return False
    class Driver:
        current_url = "https://www.facebook.com/reel/123/"
        def get(self, url): pass
        def execute_script(self, script): return False
    monkeypatch.setattr(crawler, "save_diagnostic", lambda driver, url: Path("diagnostic.json"))
    monkeypatch.setattr(crawler, "wait_until", lambda stop, condition, **kwargs: condition() or None)
    with pytest.raises(crawler.CommentLoadError, match="Chưa mở được bảng bình luận Reel"):
        crawler.crawl_post(Driver(), Driver.current_url, None, Stop(), {})


def test_single_url_ignores_missing_input(monkeypatch, tmp_path):
    url = "https://www.facebook.com/reel/123/"
    monkeypatch.setattr(sys, "argv", ["crawl_comments.py", "--url", url,
                        "--input", str(tmp_path / "missing.txt"), "--output", str(tmp_path / "out.csv")])
    monkeypatch.setattr(crawler, "read_cookies", lambda path: [])
    seen = []
    monkeypatch.setattr(crawler, "crawl_parallel", lambda urls, *args: seen.extend(urls))
    crawler.main()
    assert seen == [url]


def test_input_repairs_duplicate_scheme_skips_invalid_and_deduplicates(capsys):
    lines = ["https://https://www.facebook.com/reel/25100265262893495/",
             "https://example.com/posts/123/", "# ghi chú", "post_urls = [",
             '    "https://www.facebook.com/reel/25100265262893495/",',
             '    "https://www.facebook.com/PhimMuaDo/posts/123/",', "]"]
    assert crawler.parse_post_urls(lines) == [
        "https://www.facebook.com/reel/25100265262893495/", URL]
    log = capsys.readouterr().out
    assert "đã sửa tiền tố" in log
    assert "BỎ QUA dòng 2" in log


def test_start_line_uses_physical_lines_and_includes_selected_line(capsys):
    lines = [URL, "# ghi chú", "", "https://example.com/posts/1/",
             "https://https://www.facebook.com/reel/45/", URL]
    assert crawler.parse_post_urls(lines, start_line=4) == [
        "https://www.facebook.com/reel/45/", URL]
    log = capsys.readouterr().out
    assert "BỎ QUA dòng 4" in log
    assert "Dòng 5: đã sửa" in log
    assert crawler.parse_post_urls(lines, start_line=6) == [URL]


def test_start_line_cli_preserves_old_csv_and_workers(monkeypatch, tmp_path):
    source = tmp_path / "posts.txt"
    source.write_text(URL + "\n# ghi chú\nhttps://www.facebook.com/reel/45/\n", encoding="utf-8")
    output = tmp_path / "comments.csv"
    old = {"comment": "Đã thu", "comment_id": "old", "post_url": URL,
           "parent_comment_id": "", "collected_at": "now"}
    crawler.save_rows(output, {(URL, "old"): old})
    monkeypatch.setattr(sys, "argv", ["crawl_comments.py", "--input", str(source),
        "--output", str(output), "--start-line", "3", "--workers", "3", "--browser", "edge"])
    monkeypatch.setattr(crawler, "read_cookies", lambda path: [])
    calls = []
    def run(urls, cookies, browser, workers, max_scrolls, path, rows):
        assert urls == ["https://www.facebook.com/reel/45/"]
        assert rows[(URL, "old")] == old
        assert workers == 3 and browser == "edge"
        calls.append(True)
    monkeypatch.setattr(crawler, "crawl_parallel", run)
    crawler.main()
    assert calls == [True]
    # Số dòng vượt file phải được báo trước khi khởi chạy worker.
    monkeypatch.setattr(sys, "argv", ["crawl_comments.py", "--input", str(source), "--start-line", "4"])
    with pytest.raises(SystemExit) as error:
        crawler.main()
    assert error.value.code == 2
    assert calls == [True]


def test_failed_worker_does_not_stop_peer_or_take_jobs(monkeypatch):
    from queue import Queue
    jobs, results, stop = Queue(), Queue(), threading.Event()
    jobs.put((1, URL))
    monkeypatch.setattr(stop, "wait", lambda timeout: stop.is_set())
    def start(worker_id, browser, event):
        if worker_id == 2:
            raise WebDriverException("mock startup failure")
        return object()
    from selenium.common.exceptions import WebDriverException
    monkeypatch.setattr(crawler, "start_worker_driver", start)
    monkeypatch.setattr(crawler, "login", lambda driver, cookies: None)
    closed = []
    monkeypatch.setattr(crawler, "close_driver", lambda driver: closed.append(driver))
    def crawl(driver, url, max_scrolls, event, collected, on_progress=None):
        collected[(url, "1")] = {"comment": "Hay", "comment_id": "1", "post_url": url,
                                  "parent_comment_id": "", "collected_at": "now"}
    monkeypatch.setattr(crawler, "crawl_post", crawl)
    crawler.crawl_worker(2, jobs, results, stop, "edge", [], None)
    assert not stop.is_set()
    assert jobs.qsize() == 1
    failure = results.get_nowait()
    assert "dừng worker này" in failure[3]
    crawler.crawl_worker(1, jobs, results, stop, "edge", [], None)
    success = results.get_nowait()
    assert success[1] == URL and len(success[2]) == 1
    assert success[3] == "hoàn thành lượt cuộn"
    assert not stop.is_set() and jobs.empty() and len(closed) == 1


def test_startup_retries_once_before_failing(monkeypatch):
    from selenium.common.exceptions import WebDriverException
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda timeout: False)
    calls = []
    def make(browser):
        calls.append(browser)
        if len(calls) == 1:
            raise WebDriverException("mock temporary failure")
        return "driver"
    monkeypatch.setattr(crawler, "make_driver", make)
    assert crawler.start_worker_driver(1, "edge", stop) == "driver"
    assert calls == ["edge", "edge"]


def test_wait_returns_as_soon_as_data_arrives_and_times_out(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(crawler.time, "monotonic", lambda: clock[0])
    class Stop:
        stopped = False
        def is_set(self): return self.stopped
        def wait(self, seconds):
            clock[0] += seconds
            return self.stopped
    stop = Stop()
    calls = []
    def arriving():
        calls.append(clock[0])
        return ["new comment"] if clock[0] >= 0.6 else []
    assert crawler.wait_until(stop, arriving, timeout=3, poll=0.3) == ["new comment"]
    assert clock[0] == pytest.approx(0.6)
    clock[0] = 0
    assert crawler.wait_until(stop, lambda: [], timeout=0.7, poll=0.3) is None
    assert clock[0] == pytest.approx(0.7)
    stop.stopped = True
    assert crawler.wait_until(stop, lambda: pytest.fail("poll after stop")) is None


def test_share_links_are_accepted_as_input():
    urls = ['https://www.facebook.com/share/p/1KdZL3ZHwJ/',
            'https://www.facebook.com/share/p/1Eb3R575F2/']
    assert crawler.parse_post_urls(urls) == urls


@pytest.mark.parametrize('redirected_url,metadata', [
    (URL + '?tracking=abc', []),
    ('https://www.facebook.com/share/p/abc/', [URL]),
])
def test_share_resolution_uses_redirect_or_canonical_metadata(monkeypatch, redirected_url, metadata):
    class Driver:
        current_url = redirected_url
        def execute_script(self, script):
            return metadata if 'link[rel=' in script else False
    monkeypatch.setattr(crawler, 'wait_until', lambda stop, condition, **kwargs: condition())
    assert crawler.resolve_shared_post(Driver(), 'https://www.facebook.com/share/p/abc/', threading.Event()) == URL


def test_unresolved_share_is_error_not_empty_success(monkeypatch):
    class Driver:
        current_url = 'https://www.facebook.com/share/p/abc/'
        def execute_script(self, script):
            return [] if 'link[rel=' in script else False
    monkeypatch.setattr(crawler, 'wait_until', lambda stop, condition, **kwargs: condition())
    monkeypatch.setattr(crawler, 'save_diagnostic', lambda driver, url: Path('test-diagnostic.json'))
    with pytest.raises(crawler.CommentLoadError, match='chưa chuyển'):
        crawler.resolve_shared_post(Driver(), Driver.current_url, threading.Event())


def test_share_alias_saved_and_comments_not_duplicated(monkeypatch, tmp_path, capsys):
    import json
    share = 'https://www.facebook.com/share/p/abc/'
    monkeypatch.setattr(crawler, 'start_worker_driver', lambda *args: object())
    monkeypatch.setattr(crawler, 'login', lambda *args: None)
    monkeypatch.setattr(crawler, 'close_driver', lambda *args: None)
    def crawl(driver, url, max_scrolls, stop, collected, on_progress=None):
        row = {'comment': 'Hay', 'comment_id': '1', 'post_url': URL,
               'parent_comment_id': '', 'collected_at': 'now'}
        collected[(URL, '1')] = row
        on_progress([row])
    monkeypatch.setattr(crawler, 'crawl_post', crawl)
    output = tmp_path / 'comments.csv'
    rows = {}
    crawler.crawl_parallel([share, URL], [], 'edge', 2, 0, output, rows)
    assert len(rows) == 1
    assert json.loads(output.with_suffix('.aliases.json').read_text(encoding='utf-8')) == {share: URL}
    assert 'Đã lưu 1 bình luận' in capsys.readouterr().out
