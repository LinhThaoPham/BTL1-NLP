
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from queue import Empty, Queue
from threading import Event, Lock
import time
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.by import By

from get_post_urls import (BASE_DIR, browser_call, canonical_post_url, close_driver,
                          ensure_access, login, make_driver, positive_int, read_cookies)

COLUMNS = ["comment", "comment_id", "post_url", "parent_comment_id", "collected_at"]
DRIVER_START_LOCK = Lock()
COMMENT_SELECTOR = ('[role="article"], [data-commentid], [data-comment-id], '
                    '[aria-label^="Comment by" i], [aria-label^="Reply by" i], '
                    '[aria-label^="Bình luận của" i], [aria-label^="Phản hồi của" i]')


class CommentLoadError(ValueError):
    pass


def wait_until(stop, condition, timeout=3, poll=0.3):
    pass
    deadline = time.monotonic() + timeout
    while not stop.is_set():
        try:
            result = condition()
            if result:
                return result
        except StaleElementReferenceException:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0 or stop.wait(min(poll, remaining)):
            break
    return None


def comment_container(node):
    return node.find_parent(lambda tag: tag.get("role") == "article"
                            or tag.has_attr("data-commentid") or tag.has_attr("data-comment-id")
                            or re.match(r"comment by|reply by|bình luận của|phản hồi của",
                                        tag.get("aria-label", ""), re.I))


def identity(article):
    own = article.get("data-commentid") or article.get("data-comment-id") or ""
    parent = article.get("data-parent-comment-id", "")
    for anchor in article.select("a[href]"):
        if comment_container(anchor) is not article:
            continue
        query = parse_qs(urlparse(anchor["href"]).query)
        if query.get("comment_id"):
            own = query.get("reply_comment_id", query["comment_id"])[0]
            
            parent = parent or (query["comment_id"][0] if query.get("reply_comment_id") else "")
            break
    return str(own), str(parent)


def extract_comments(html, post_url):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for article in soup.select(COMMENT_SELECTOR):
        comment_id, parent_id = identity(article)
        label = article.get("aria-label", "")
        if not comment_id and not re.match(r"comment by|reply by|bình luận của|phản hồi của", label, re.I):
            continue
        if not parent_id:
            parent = comment_container(article)
            while parent is not None:
                parent_id = identity(parent)[0]
                if parent_id:
                    break
                parent = comment_container(parent)
        
        copy = BeautifulSoup(str(article), "html.parser").select_one(COMMENT_SELECTOR)
        for nested in list(copy.select(COMMENT_SELECTOR)):
            if nested.parent is not None:
                nested.decompose()
        blocks = copy.select('[data-ad-rendering-role="comment_message"]')
        if not blocks:
            blocks = [node for node in copy.select('[dir="auto"]')
                      if not node.find_parent(["a", "button"])
                      and not node.find_parent(attrs={"role": "button"})
                      and not node.select('[dir="auto"]')]
        
        for block in blocks:
            for image in block.select('img[alt]'):
                image.replace_with(image.get("alt", ""))
        text = " ".join(node.get_text(" ", strip=True) for node in blocks).strip()
        if not text:
            continue
        
        if not comment_id:
            payload = f"{post_url}|{label}|{text}"
            comment_id = "fallback:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        rows.append({"comment": text, "comment_id": comment_id, "post_url": post_url,
                     "parent_comment_id": parent_id,
                     "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return rows



COMMENTS_READY_SCRIPT = """
    const visible = el => el.getClientRects().length > 0 &&
        getComputedStyle(el).visibility !== 'hidden';
    const nodes = document.querySelectorAll(
        '[data-ad-rendering-role="comment_message"], [data-commentid], [data-comment-id], '
        + '[aria-label^="Comment by" i], [aria-label^="Bình luận của" i]');
    if (Array.from(nodes).some(visible)) return true;
    return Array.from(document.querySelectorAll('[contenteditable="true"], textarea')).some(el =>
        visible(el) && /bình luận|comment/i.test(
            [el.getAttribute('aria-label'), el.getAttribute('data-placeholder'),
             el.getAttribute('placeholder')].filter(Boolean).join(' ')));
"""

OPEN_COMMENTS_SCRIPT = """
    const visible = el => el.getClientRects().length > 0 &&
        getComputedStyle(el).visibility !== 'hidden';
    const buttons = Array.from(document.querySelectorAll('[role="button"], button, a[role="link"]'));
    const candidates = [];
    for (const el of buttons) {
        if (!visible(el) || el.closest('[role="menu"], [data-commentid], [data-comment-id]')) continue;
        const label = [el.getAttribute('aria-label'), el.getAttribute('title'), el.innerText,
            ...Array.from(el.querySelectorAll('[aria-label]'), n => n.getAttribute('aria-label'))
        ].filter(Boolean).join(' ').replace(/\\s+/g, ' ').trim();
        if (!/bình luận|comments?/i.test(label) || label.length > 180) continue;
        if (/viết|write|reply|phản hồi|chia sẻ|share|thích|like/i.test(label)) continue;
        candidates.push({el, score: (/\\d/.test(label) ? 5 : 0) +
            (/^(bình luận|comments?)(\\s|$)/i.test(label) ? 3 : 0)});
    }
    candidates.sort((a,b) => b.score - a.score);
    return candidates.length ? candidates[0].el : null;
"""


def open_comments_panel(driver, url, stop):
    if browser_call(driver, lambda: driver.execute_script(COMMENTS_READY_SCRIPT)):
        return True
    button = browser_call(driver, lambda: driver.execute_script(OPEN_COMMENTS_SCRIPT))
    if not button:
        return False
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
    button.click()
    print(f"Đã bấm mở bảng bình luận: {url}")
    def ready():
        ensure_access(driver)
        return browser_call(driver, lambda: driver.execute_script(COMMENTS_READY_SCRIPT))
    return bool(wait_until(stop, ready, timeout=5))


def save_diagnostic(driver, url):
    pass
    folder = BASE_DIR / "diagnostics"
    folder.mkdir(exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()[:12] + "_" + time.strftime("%Y%m%d_%H%M%S")
    path = folder / name
    info = {"url": url}
    try:
        driver.save_screenshot(str(path.with_suffix(".png")))
        info["screenshot"] = str(path.with_suffix(".png"))
    except (WebDriverException, AttributeError):
        pass
    try:
        info["dom"] = driver.execute_script("""
            return {articles: document.querySelectorAll('[role="article"]').length,
                    comment_messages: document.querySelectorAll('[data-ad-rendering-role="comment_message"]').length,
                    dialogs: document.querySelectorAll('[role="dialog"]').length};
        """)
    except (WebDriverException, AttributeError):
        pass
    path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.with_suffix(".json")


def expand_comments(driver):
    pattern = re.compile(
        r"^(?:xem thêm|xem bình luận trước|xem thêm bình luận|xem phản hồi|xem tất cả.*phản hồi|"
        r"view more|view previous comments|view more comments|view replies|see more|"
        r"view \d+ more repl(?:y|ies)|\d+ phản hồi|xem thêm \d+ bình luận|"
        r"xem \d+ phản hồi|view \d+ more comments|"
        r"(?:xem|hiển thị|view|see|show)\s+(?:(?:tất cả|all|thêm|more)\s+)?"
        r"(?:[\d.,]+\s+)?(?:(?:thêm|more)\s+)?(?:phản hồi|câu trả lời|replies|reply))"
        r"(?:\s.*)?$", re.I)
    clicked = 0
    for button in driver.find_elements(By.CSS_SELECTOR, '[role="button"], button'):
        try:
            label = button.text.strip() or (button.get_attribute("aria-label") or "").strip()
            if pattern.fullmatch(label) and button.is_displayed() and button.is_enabled():
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
                button.click()
                clicked += 1
                if clicked >= 20:
                    break
        except WebDriverException:
            continue
    return clicked


FILTER_CONTROL_SCRIPT = """
    const visible = el => el.getClientRects().length > 0 &&
        getComputedStyle(el).visibility !== 'hidden';
    const dialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
    const root = dialogs.reverse().find(el => el.querySelector('[role="article"], [data-commentid], [data-comment-id]')) || document;
    const labels = ['phù hợp nhất', 'most relevant', 'mới nhất', 'newest',
                    'tất cả bình luận', 'all comments'];
    for (const el of root.querySelectorAll('span, [role="button"], button, [aria-haspopup]')) {
        if (!visible(el) || el.closest('[role="menu"], [role="menuitem"], [role="menuitemradio"], [role="option"]')) continue;
        const label = (el.innerText || '').replace(/\\s+/g, ' ').trim().toLowerCase();
        if (!labels.includes(label)) continue;
        const button = el.closest('[role="button"], button, [aria-haspopup], [tabindex="0"]');
        if (button && visible(button)) return {element: button, label: label};
    }
    return null;
"""

ALL_COMMENTS_OPTION_SCRIPT = """
    const visible = el => el.getClientRects().length > 0 &&
        getComputedStyle(el).visibility !== 'hidden';
    // Facebook có thể đặt chữ trong span và mục menu trong role=button.
    for (const el of document.querySelectorAll('span, [role="menuitem"], [role="menuitemradio"], [role="option"], [role="button"]')) {
        if (!visible(el)) continue;
        const label = (el.innerText || '').replace(/\\s+/g, ' ').trim().toLowerCase();
        if (label !== 'tất cả bình luận' && label !== 'all comments') continue;
        const option = el.closest('[role="menuitem"], [role="menuitemradio"], [role="option"], [role="button"], [tabindex="0"]');
        if (option && visible(option)) return option;
    }
    return null;
"""


def select_all_comments(driver, stop=None):
    from selenium.webdriver.common.keys import Keys
    stop = stop if stop is not None else Event()
    for attempt in range(3):
        if stop.is_set():
            return False
        ensure_access(driver)
        try:
            control = wait_until(stop, lambda: driver.execute_script(FILTER_CONTROL_SCRIPT), timeout=1)
            if not control:
                continue
            if control["label"] in {"tất cả bình luận", "all comments"}:
                return True
            control["element"].click()
            option = wait_until(stop, lambda: driver.execute_script(ALL_COMMENTS_OPTION_SCRIPT), timeout=2)
            if option:
                option.click()
                def selected():
                    control = driver.execute_script(FILTER_CONTROL_SCRIPT)
                    return control and control["label"] in {"tất cả bình luận", "all comments"}
                if wait_until(stop, selected, timeout=2):
                    return True
            driver.switch_to.active_element.send_keys(Keys.ESCAPE)
        except WebDriverException:
            continue
    return False


SCROLL_COMMENTS_SCRIPT = """
    const visible = el => el.getClientRects().length > 0 &&
        getComputedStyle(el).visibility !== 'hidden';
    const markers = '[data-ad-rendering-role="comment_message"], [data-commentid], [data-comment-id], '
        + '[aria-label^="Comment by" i], [aria-label^="Bình luận của" i], [role="article"]';
    const dialogs = Array.from(document.querySelectorAll('[role="dialog"], [role="complementary"]')).filter(visible);
    const root = dialogs.reverse().find(el => el.querySelector(markers)) || document;
    const articles = Array.from(root.querySelectorAll(markers)).filter(visible);
    let target = null;
    // Tìm tổ tiên cuộn được của bình luận cuối; bài có thể nằm trong popup.
    for (let el = articles[articles.length - 1]; el; el = el.parentElement) {
        if (el.scrollHeight > el.clientHeight + 5 && el.clientHeight > 100 &&
            /auto|scroll/.test(getComputedStyle(el).overflowY)) { target = el; break; }
    }
    if (!target && root !== document) {
        target = [root, ...root.querySelectorAll('*')].find(el => visible(el) &&
            el.clientHeight > 100 && el.scrollHeight > el.clientHeight + 5 &&
            /auto|scroll/.test(getComputedStyle(el).overflowY));
    }
    target = target || document.scrollingElement;
    const before = target.scrollTop;
    target.scrollBy(0, Math.max(target.clientHeight * 0.8, 400));
    return {at_bottom: target.scrollTop + target.clientHeight >= target.scrollHeight - 30,
            moved: target.scrollTop !== before};
"""


def save_rows(path, rows):
    pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())
    os.replace(temporary, path)


def crawl_post(driver, url, max_scrolls, stop, collected, on_progress=None):
    browser_call(driver, lambda: driver.get(url))
    wait_until(stop, lambda: driver.execute_script(COMMENTS_READY_SCRIPT)
               or driver.execute_script(OPEN_COMMENTS_SCRIPT), timeout=5)
    if stop.is_set():
        return
    ensure_access(driver)
    panel_ready = open_comments_panel(driver, url, stop)
    if stop.is_set():
        return
    if not panel_ready and "/reel/" in urlparse(url).path:
        diagnostic = save_diagnostic(driver, url)
        raise CommentLoadError(f"Chưa mở được bảng bình luận Reel; không phải bài có 0 bình luận. Thông tin kiểm tra: {diagnostic}")
    if not select_all_comments(driver, stop):
        print(f"CẢNH BÁO: chưa chọn được Tất cả bình luận, dữ liệu có thể thiếu: {url}")
    else:
        print(f"Đã xác nhận bộ lọc Tất cả bình luận: {url}")
    turn, stagnant = 0, 0
    previous = None
    at_bottom = False
    while True:
        if stop.is_set():
            return
        ensure_access(driver)
        html = browser_call(driver, lambda: driver.page_source)
        for row in extract_comments(html, url):
            collected[(url, row["comment_id"])] = row
        snapshot = {key: row["comment"] for key, row in collected.items()}
        changed = snapshot != previous
        if changed and collected and on_progress:
            on_progress(list(collected.values()))
        
        stagnant = stagnant + 1 if at_bottom and not changed else 0
        previous = snapshot
        print(f"Bài {url}: lượt {turn}, {len(collected)} bình luận, chờ cuối trang {stagnant}/3")
        if stagnant >= 3:
            if not collected:
                diagnostic = save_diagnostic(driver, url)
                raise CommentLoadError(f"Không nhận diện được bình luận; chưa xác nhận bài không có bình luận. Thông tin kiểm tra: {diagnostic}")
            print(f"Không tải thêm được bình luận sau 3 lượt: {url}")
            return
        if max_scrolls is not None and turn >= max_scrolls:
            print(f"Dừng theo --max-scrolls, chưa xác nhận hết bình luận: {url}")
            return
        expand_comments(driver)
        ensure_access(driver)
        position = browser_call(driver, lambda: driver.execute_script(SCROLL_COMMENTS_SCRIPT))
        at_bottom = position["at_bottom"]
        def new_comments():
            ensure_access(driver)
            found = extract_comments(driver.page_source, url)
            return [row for row in found if (url, row["comment_id"]) not in collected
                    or collected[(url, row["comment_id"])]["comment"] != row["comment"]]
        new_rows = wait_until(stop, new_comments, timeout=3)
        if new_rows:
            for row in new_rows:
                collected[(url, row["comment_id"])] = row
        if stop.is_set():
            return
        turn += 1


def start_worker_driver(worker_id, browser, stop):
    for attempt in range(2):
        if stop.is_set():
            return None
        try:
            
            
            with DRIVER_START_LOCK:
                if stop.is_set():
                    return None
                print(f"[Worker {worker_id}] Đang mở {browser}, lần {attempt + 1}/2...", flush=True)
                driver = make_driver(browser)
            print(f"[Worker {worker_id}] Đã mở trình duyệt riêng.", flush=True)
            return driver
        except Exception as error:
            
            message = str(error).split("Stacktrace:")[0].strip()[:500]
            print(f"[Worker {worker_id}] Lỗi mở trình duyệt ({type(error).__name__}): {message}", flush=True)
            if attempt == 1:
                raise
            if stop.wait(2):
                return None


def crawl_worker(worker_id, jobs, results, stop, browser, cookies, max_scrolls):
    
    driver = None
    try:
        if stop.is_set():
            return
        driver = start_worker_driver(worker_id, browser, stop)
        if driver is None:
            return
        login(driver, cookies)
        print(f"[Worker {worker_id}] Đã nạp phiên, bắt đầu nhận bài.", flush=True)
        while not stop.is_set():
            try:
                index, url = jobs.get_nowait()
            except Empty:
                break
            collected = {}
            status = "hoàn thành lượt cuộn"
            retire = False
            try:
                print(f"[Worker {worker_id}] Bài {index}: {url}")
                crawl_post(driver, url, max_scrolls, stop, collected,
                           on_progress=lambda comments: results.put(
                               (worker_id, url, comments, "đang thu, đã lưu tiến độ", False)))
                if stop.is_set():
                    status = "đã dừng, lưu phần thu được"
            except RuntimeError as error:
                status = f"{error}; dừng worker này"
                retire = True
            except CommentLoadError as error:
                status = str(error)
            except Exception as error:
                status = f"lỗi {type(error).__name__}; lưu phần thu được, bài này cần chạy lại"
            finally:
                results.put((worker_id, url, list(collected.values()), status, True))
                jobs.task_done()
            if retire:
                break
    except Exception as error:
        
        detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        results.put((worker_id, "", [], f"không khởi tạo/đăng nhập được ({detail}); dừng worker này, các worker khác tiếp tục", True))
    finally:
        if driver is not None:
            close_driver(driver)


def crawl_parallel(urls, cookies, browser, workers, max_scrolls, output, rows):
    jobs, results, stop = Queue(), Queue(), Event()
    for index, url in enumerate(urls, 1):
        jobs.put((index, url))
    worker_count = min(workers, len(urls))
    print(f"Yêu cầu {workers} worker, có {len(urls)} URL cần xử lý: chạy {worker_count} worker, mỗi worker một trình duyệt riêng.", flush=True)
    finished = 0

    def store_result(result):
        nonlocal finished
        worker_id, url, comments, status, final = result
        before = len(rows)
        for row in comments:
            rows[(row["post_url"], row["comment_id"])] = row
        
        save_rows(output, rows)
        if url and final:
            finished += 1
            if status == "hoàn thành lượt cuộn":
                post_urls = {row["post_url"] for row in comments} or {url}
                post_count = sum(row["post_url"] in post_urls for row in rows.values())
                print(
                    f"[Worker {worker_id}] XONG POST: {url}\n"
                    f"  Đã lưu {post_count} bình luận của bài vào {output}. "
                    f"Đã xử lý {finished}/{len(urls)} bài.",
                    flush=True,
                )
            else:
                print(f"[Worker {worker_id}] POST LỖI/DỪNG: {url}\n"
                      f"  {status}. Đã lưu phần thu được; đã xử lý {finished}/{len(urls)} bài.",
                      flush=True)
        else:
            print(f"[Worker {worker_id}] {status}: {url}; thêm {len(rows) - before}, tổng {len(rows)} bình luận.",
                  flush=True)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(crawl_worker, i + 1, jobs, results, stop,
                                   browser, cookies, max_scrolls) for i in range(worker_count)]
        try:
            while not all(future.done() for future in futures) or not results.empty():
                try:
                    store_result(results.get(timeout=0.5))
                except Empty:
                    continue
        except KeyboardInterrupt:
            print("Đang dừng worker và lưu dữ liệu đã thu được; vui lòng chờ đóng trình duyệt...")
            stop.set()
        finally:
            stop.set()
            
            for future in futures:
                future.result()
            while not results.empty():
                store_result(results.get_nowait())
    print(f"Đã xử lý {finished}/{len(urls)} bài. Các bài bị lỗi/dừng cần chạy lại.")


def parse_post_urls(lines, start_line=1):
    urls, seen = [], set()
    for line_number, line in enumerate(lines, 1):
        if line_number < start_line:
            continue
        value = line.strip().lstrip("\ufeff")
        if not value or value.startswith("#"):
            continue
        
        if value in {"post_urls = [", "post_urls=[", "[", "]", "];"}:
            continue
        value = value.rstrip(",").strip().strip("\"'")
        corrected = re.sub(r"^(?:https?://){2,}", "https://", value, flags=re.I)
        if corrected != value:
            print(f"Dòng {line_number}: đã sửa tiền tố http(s):// bị lặp.")
        try:
            url = canonical_post_url(corrected)
        except ValueError as error:
            print(f"BỎ QUA dòng {line_number}: {error}")
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def main():
    parser = argparse.ArgumentParser(description="Thu bình luận từ danh sách bài Facebook.")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "post_urls.txt")
    parser.add_argument("--start-line", type=positive_int, default=1,
                        help="Bắt đầu từ dòng này trong file --input, gồm dòng đó (mặc định: 1)")
    parser.add_argument("--url", help="Thử riêng một bài/Reel, bỏ qua danh sách --input")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "facebook_comments_all.csv")
    parser.add_argument("--max-scrolls", type=positive_int, default=None,
                        help="Giới hạn lượt cuộn mỗi bài (mặc định: cuộn đến khi không tải thêm được)")
    parser.add_argument("--workers", type=positive_int, choices=range(1, 6), default=1,
                        help="Số trình duyệt chạy cùng lúc, từ 1 đến 5 (mặc định: 1)")
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome")
    parser.add_argument("--cookies", type=Path, default=BASE_DIR / "cookies.json")
    args = parser.parse_args()
    if args.url and args.start_line != 1:
        parser.error("--start-line dùng với file --input; không dùng cùng --url.")
    if not args.url and not args.input.exists():
        parser.error("Chưa có post_urls.txt. Chạy get_post_urls.py trước.")
    lines = [args.url] if args.url else args.input.read_text(encoding="utf-8-sig").splitlines()
    if args.start_line > len(lines):
        parser.error(f"File chỉ có {len(lines)} dòng, không thể bắt đầu từ dòng {args.start_line}.")
    urls = parse_post_urls(lines, start_line=args.start_line)
    if not urls:
        parser.error(f"Không có URL bài viết/Reel hợp lệ từ dòng {args.start_line} trở đi.")
    if not args.url:
        print(f"Đọc {args.input} từ dòng {args.start_line} đến hết: {len(urls)} URL hợp lệ, không trùng.")
    rows = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS):
                parser.error("CSV cũ khác cấu trúc. Chọn đường dẫn --output mới.")
            rows = {(row["post_url"], row["comment_id"]): row for row in reader}
    try:
        cookies = read_cookies(args.cookies)
    except ValueError as error:
        parser.error(str(error))
    crawl_parallel(urls, cookies, args.browser, args.workers, args.max_scrolls, args.output, rows)
    print(f"File kết quả: {args.output}")
    print("Số lượt cuộn không bảo đảm lấy đủ bình luận. Kiểm tra CSV và chế độ lọc bình luận trên Facebook.")


if __name__ == "__main__":
    main()
