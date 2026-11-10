
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By

BASE_DIR = Path(__file__).resolve().parent
PAGE_URL = "https://www.facebook.com/PhimMuaDo"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Giá trị phải lớn hơn 0.")
    return number


def facebook_url(value):
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "http"} or parsed.hostname not in {
        "facebook.com", "www.facebook.com", "m.facebook.com", "mbasic.facebook.com"
    }:
        raise ValueError(f"Không phải URL Facebook hợp lệ: {value}")
    return parsed


def canonical_post_url(value):
    pass
    parsed = facebook_url(urljoin("https://www.facebook.com/", value))
    parts = parsed.path.strip("/").split("/")
    query = parse_qs(parsed.query)
    for section in ("posts", "videos", "permalink", "reel"):
        if section in parts:
            index = parts.index(section)
            if index + 1 < len(parts) and parts[index + 1]:
                return "https://www.facebook.com/" + "/".join(parts[:index + 2]) + "/"
    if query.get("story_fbid") and query.get("id"):
        return "https://www.facebook.com/permalink.php?" + urlencode({
            "story_fbid": query["story_fbid"][0], "id": query["id"][0]
        })
    if parsed.path.rstrip("/") == "/watch" and query.get("v"):
        return "https://www.facebook.com/videos/" + query["v"][0] + "/"
    if parsed.path.rstrip("/") in {"/photo", "/photo.php"} and query.get("fbid"):
        return "https://www.facebook.com/photo.php?" + urlencode({"fbid": query["fbid"][0]})
    raise ValueError(f"Chưa nhận diện được link bài viết: {value}")


def make_driver(browser="chrome"):
    cache = BASE_DIR / ".selenium_cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("SE_CACHE_PATH", str(cache))
    os.environ.setdefault("SE_AVOID_STATS", "true")
    options = webdriver.EdgeOptions() if browser == "edge" else webdriver.ChromeOptions()
    
    options.page_load_strategy = "eager"
    options.add_argument("--disable-notifications")
    options.add_argument("--lang=vi-VN")
    options.add_argument("--window-size=1440,1000")
    factory = webdriver.Edge if browser == "edge" else webdriver.Chrome
    driver = factory(options=options)
    driver.set_page_load_timeout(120)
    driver.set_script_timeout(30)
    return driver


def browser_call(driver, action, attempts=3):
    pass
    for attempt in range(attempts):
        try:
            return action()
        except TimeoutException:
            if attempt + 1 == attempts:
                raise
            print(f"Trình duyệt phản hồi chậm; đang thử lại {attempt + 1}/{attempts - 1}...")
            try:
                driver.execute_script("window.stop();")
            except WebDriverException:
                pass
            time.sleep(3)


def close_driver(driver):
    try:
        driver.quit()
    except WebDriverException:
        pass


def read_cookies(path):
    if not path.exists():
        raise ValueError(f"Chưa có file cookie: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError):
        raise ValueError("File cookie phải là JSON hợp lệ.") from None
    
    if isinstance(data, dict):
        data = data.get("cookies")
    if not isinstance(data, list):
        raise ValueError("Cookie phải là danh sách các đối tượng name/value.")
    cookies = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("value"), str):
            raise ValueError("Mỗi cookie cần name và value dạng chuỗi.")
        if not item["value"].strip():
            continue
        
        domain = item.get("domain", ".facebook.com")
        if not isinstance(domain, str) or domain.lstrip(".").lower() not in {
            "facebook.com", "www.facebook.com", "m.facebook.com", "mbasic.facebook.com"
        }:
            raise ValueError("Chỉ nạp cookie thuộc Facebook.")
        cookie = {"name": item["name"], "value": item["value"], "path": "/",
                  "domain": ".facebook.com", "secure": True}
        if isinstance(item.get("httpOnly"), bool):
            cookie["httpOnly"] = item["httpOnly"]
        cookies.append(cookie)
    if not {"c_user", "xs"}.issubset({item["name"] for item in cookies}):
        raise ValueError("Điền giá trị cookie c_user và xs trong cookies.json trước khi chạy.")
    return cookies


def login(driver, cookies):
    browser_call(driver, lambda: driver.get("https://www.facebook.com/"))
    driver.delete_all_cookies()
    for cookie in cookies:
        try:
            driver.add_cookie(cookie)
        except Exception:
            
            raise RuntimeError("Không nạp được cookie vào trình duyệt. Kiểm tra file cookie.") from None
    browser_call(driver, lambda: driver.get("https://www.facebook.com/"))
    time.sleep(5)
    ensure_access(driver)
    print("Đã nạp cookie; bắt đầu truy cập nguồn.")


def ensure_access(driver):
    path = urlparse(browser_call(driver, lambda: driver.current_url)).path.lower()
    if any(marker in path for marker in ("/login", "/checkpoint", "/challenge")):
        raise RuntimeError("Facebook yêu cầu đăng nhập/xác thực. Kiểm tra phiên và cập nhật cookie rồi chạy lại.")
    
    login_visible = browser_call(driver, lambda: driver.execute_script("""
        return Array.from(document.querySelectorAll(
            'input[type="password"], input[name="pass"], input[name="email"]'
        )).some(el => el.getClientRects().length > 0 &&
            getComputedStyle(el).visibility !== 'hidden');
    """))
    if login_visible:
        raise RuntimeError("Facebook hiện hộp đăng nhập trên trang. Cookie chưa được chấp nhận hoặc đã hết hạn; cập nhật cookies.json rồi chạy lại.")


def get_group_posts(driver, url, max_scrolls=None, max_posts=None, wait=3,
                    stagnant_rounds=10, on_progress=None):
    pass
    facebook_url(url)
    browser_call(driver, lambda: driver.get(url))
    time.sleep(wait)
    links, seen = [], set()
    turn, stagnant = 0, 0
    while True:
        ensure_access(driver)
        before = len(links)
        
        hrefs = browser_call(driver, lambda: driver.execute_script(
            "return Array.from(document.querySelectorAll('a[href]'), a => a.href)"
            ".filter(h => /\\/(posts|videos|permalink|reel)\\/|[?&]story_fbid=/.test(h));"))
        for href in hrefs:
            try:
                if not href:
                    continue
                clean = canonical_post_url(href)
            except ValueError:
                continue
            if clean not in seen:
                seen.add(clean)
                links.append(clean)
                if max_posts is not None and len(links) >= max_posts:
                    break
        if len(links) > before and on_progress:
            on_progress(links)
        print(f"Lượt {turn}: {len(links)} link (+{len(links) - before})")
        if max_posts is not None and len(links) >= max_posts:
            print("Đạt giới hạn --max-posts đã chọn.")
            break
        if max_scrolls is not None and turn >= max_scrolls:
            print("Đạt giới hạn --max-scrolls đã chọn.")
            break
        at_bottom = browser_call(driver, lambda: driver.execute_script(
            "const el = document.scrollingElement; "
            "return el.scrollTop + el.clientHeight >= el.scrollHeight - 100;"))
        stagnant = stagnant + 1 if at_bottom and len(links) == before else 0
        if stagnant >= stagnant_rounds:
            print(f"Dừng: {stagnant_rounds} lượt ở cuối trang không tìm thấy link mới.")
            break
        browser_call(driver, lambda: driver.execute_script("window.scrollBy(0, Math.max(window.innerHeight * 0.8, 600));"))
        time.sleep(wait)
        turn += 1
    return links


def main():
    parser = argparse.ArgumentParser(description="Lấy link bài viết từ fanpage/group Facebook.")
    parser.add_argument("--url", default=PAGE_URL, help="URL fanpage hoặc group (mặc định: fanpage Phim Mưa Đỏ)")
    parser.add_argument("--max-posts", type=positive_int, default=None, help="Giới hạn link (mặc định: không giới hạn)")
    parser.add_argument("--max-scrolls", type=positive_int, default=None, help="Giới hạn lượt cuộn (mặc định: không giới hạn)")
    parser.add_argument("--stagnant-rounds", type=positive_int, default=10, help="Số lượt liên tiếp ở cuối trang không có link mới để dừng")
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome")
    parser.add_argument("--cookies", type=Path, default=BASE_DIR / "cookies.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "post_urls.txt")
    args = parser.parse_args()
    facebook_url(args.url)
    try:
        cookies = read_cookies(args.cookies)
    except ValueError as error:
        parser.error(str(error))
    old = args.output.read_text(encoding="utf-8-sig").splitlines() if args.output.exists() else []
    saved = list(dict.fromkeys(line.strip() for line in old if line.strip()))
    def save_progress(links):
        merged = list(dict.fromkeys(saved + links))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text("".join(url + "\n" for url in merged), encoding="utf-8")
        temporary.replace(args.output)
        saved[:] = merged

    driver = make_driver(args.browser)
    try:
        login(driver, cookies)
        links = get_group_posts(driver, args.url, args.max_scrolls, args.max_posts,
                                stagnant_rounds=args.stagnant_rounds, on_progress=save_progress)
        save_progress(links)
        print(f"Thu được {len(links)} link; file có tổng cộng {len(saved)} link: {args.output}")
    except KeyboardInterrupt:
        print(f"Đã dừng theo yêu cầu. Các lượt đã thu được lưu tại: {args.output}")
    except TimeoutException:
        parser.exit(1, f"Trình duyệt vẫn không phản hồi sau 3 lần thử. Link đã lưu được giữ tại {args.output}. Đóng cửa sổ Selenium và chạy lại.\n")
    finally:
        close_driver(driver)


if __name__ == "__main__":
    main()
