"""
Twitter15/16 Tweet Content Hydration Scraper
================================================
用途：讀取 Twitter15/16 tree 檔案中出現過的 tweet_id，逐條用 Selenium
造訪 X 網頁版抓取原文、時間戳、互動數，並以 JSONL 格式增量寫入，
支援斷點續爬（中斷後重跑會自動跳過已完成的 ID）。

使用前請注意：
1. 本腳本刻意不包含任何驗證碼 / 反機器人機制的自動繞過。遇到驗證
   挑戰或登入牆時，不會暫停等人工輸入——因為這種畫面通常代表整個
   帳號/session 的行為被判定為異常，不是單一頁面的問題，重試同一
   頁面沒有意義。腳本會就地重新整理一次確認不是誤判，還是被擋的話
   就整個丟給外層重建瀏覽器、拉長冷卻時間再自動重試（見 run_forever
   的 SessionBlockedException 處理），這條 tweet_id 不會被標記完成，
   下一輪會自動排回待爬清單，可以放著整晚跑，不需要人在旁邊按 Enter。
2. 建議先用 --limit 跑一小批（例如 20 條），確認欄位有正確抓到、
   選擇器沒有失效，再放大規模跑。
3. X 的前端 DOM 結構會不定期調整，下面用的 data-testid 選擇器是
   目前普遍可用的寫法，但無法保證長期有效；如果抓回來的欄位大量是
   None，先用瀏覽器開發者工具檢查對應元素的選擇器是否變了。
4. 建議評估是否要用你的主要個人帳號登入的 profile 來跑——自動化行為
   有可能導致帳號被限制，用小號 profile 風險較低。
5. 使用前請自行確認這是否符合 X 服務條款、你系所的學術倫理規範。
6. 已設定 page_load_timeout，加上外層無窮重試迴圈：任何導致整個
   session 掛掉的例外（網路卡死、瀏覽器崩潰、驗證畫面）都會被攔下
   來，重建瀏覽器後從進度檔記錄的位置繼續。若要真正停止，用 Ctrl+C。
7. 轉推偵測邏輯是雙重判斷：(a) 網址是否還是原本要求的 tweet_id
   （對「導向到別的網址」這種轉推有效）；(b) 頁面上有沒有出現
   「XX 轉發了」這種 social context 提示（對「網址不變、原地換成
   原推文內容」這種轉推有效，這是實測發現比較常見的情況，(a) 單獨
   抓不到）。兩種只要中一種就標記成轉推，不繼續當成乾淨資料存。
8. 已知這個 social context 的 CSS selector（data-testid="socialContext"）
   是憑常見寫法猜的，沒辦法在這裡實際打開 X 驗證。跑一小批後回頭
   看 redirected_possible_retweet 的數量，如果還是接近 0、但用資料
   健檢腳本查出來的重複污染比例仍然很高，代表這個 selector 猜錯了，
   需要你截一張確定是轉推的頁面畫面，用開發者工具找實際的元素。

輸出的 status 欄位可能值：
- ok                        ：正常抓到內容
- deleted_or_unavailable    ：貼文已刪除或不存在
- suspended_account         ：貼文所屬帳號已被停權
- protected_account         ：貼文所屬帳號設為保護（非公開）
- redirected_possible_retweet：這個 tweet_id 疑似是轉推（網址導向或
                                social context 提示任一種判斷成立），
                                username/text 不可信任為此節點本身的
                                內容，redirected_to 欄位會記錄實際
                                導向的網址（原地顯示的情況這欄會是 None）
- timeout                   ：等了 wait_seconds 秒仍找不到
                                article[data-testid="tweet"]，或整個頁面
                                載入超過 page_load_timeout，可能是未預期
                                的頁面狀態或網路問題，建議抽樣人工檢查
- unexpected_error: ...     ：非預期例外（防止單一 ID 出錯拖垮整批）
- webdriver_error: ...      ：瀏覽器層級的例外，訊息會附在後面

（另外：遇到驗證挑戰/登入牆的 tweet_id 不會寫入 progress_file，
不會出現在上面這個 status 清單裡——因為這種情況會讓整個 session
重建，這條會自動留在待爬清單裡等下一輪重試，不是抓取失敗的記錄。）
"""

import os
import json
import time
import random
import argparse
from ast import literal_eval

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException
)
from webdriver_manager.firefox import GeckoDriverManager


PAGE_LOAD_TIMEOUT = 30  # 秒，超過就丟 TimeoutException，避免卡死不回應


class SessionBlockedException(Exception):
    """
    偵測到驗證挑戰 / 登入牆等需要人工處理的畫面。
    刻意不在這裡自動繞過，往外拋出交給 run_forever 重建瀏覽器、
    拉長冷卻時間後重試，讓整個腳本可以無人值守繼續跑。
    """
    pass


# ---------------------------------------------------------------------------
# Step A: 從 tree 檔案解析出所有需要爬的 tweet_id（去重）
# ---------------------------------------------------------------------------

def extract_ids_from_tree_dir(tree_dir):
    """解析 twitter15/16 的 tree/*.txt，回傳所有出現過的 tweet_id 去重集合。

    每行格式類似：
        ['uid1', 'tweetid1', '0.0']->['uid2', 'tweetid2', '5.2']
    """
    id_set = set()
    for fname in os.listdir(tree_dir):
        path = os.path.join(tree_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "->" not in line:
                    continue
                for chunk in line.split("->"):
                    try:
                        _uid, tweet_id, _t = literal_eval(chunk.strip())
                    except (ValueError, SyntaxError):
                        continue
                    if tweet_id and tweet_id != "ROOT":
                        id_set.add(tweet_id)
    return id_set


def load_priority_ids(label_path):
    """
    從 label.txt（格式 label:eid）取出所有來源推文 id。
    這些是每個事件的第一則推文，數量遠小於整個 tree 的節點數，
    優先爬完這批可以快速補齊標籤分布不平衡的問題，不用等整個
    tree 爬到 100% 才有平衡的資料可用。
    """
    priority_ids = set()
    if not label_path or not os.path.exists(label_path):
        return priority_ids
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            _label, eid = line.split(":", 1)
            priority_ids.add(eid)
    return priority_ids


# ---------------------------------------------------------------------------
# Step B: 續爬進度管理
# ---------------------------------------------------------------------------

def load_done_ids(progress_file):
    done = set()
    if os.path.exists(progress_file):
        with open(progress_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["tweet_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ---------------------------------------------------------------------------
# Step C: 建立瀏覽器（沿用你附檔的做法：帶入你自己登入過的 profile）
# ---------------------------------------------------------------------------

def build_driver(profile_path):
    options = Options()
    options.add_argument("--profile")
    options.add_argument(profile_path)
    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options,
    )
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


# ---------------------------------------------------------------------------
# Step D: 偵測驗證挑戰 / 登入牆（不自動繞過，只負責偵測）
# ---------------------------------------------------------------------------

def detect_block(driver):
    """
    回傳偵測到的封鎖類型字串（'recaptcha' / 'login_wall' / 'error_page'），
    沒有偵測到則回傳 None。

    只承認「真的有顯示出來」的元素（is_displayed()），不是只要文字
    出現在 DOM 裡就算——X 的頁面模板常常藏著預留給錯誤狀態用的隱藏
    文字節點，即使頁面其實是正常載入中，單純比對文字存在會誤判。
    """
    blocked_markers = [
        ("recaptcha", (By.XPATH, "//iframe[contains(@title, 'recaptcha')]")),
        ("login_wall", (By.XPATH, "//*[contains(text(), 'Enter your phone number or email')]")),
        ("login_wall", (By.XPATH, "//*[contains(text(), '輸入你的電話號碼或電子郵件')]")),
        ("error_page", (By.XPATH, "//*[contains(text(), 'Something went wrong')]")),
    ]
    for label, locator in blocked_markers:
        try:
            el = driver.find_element(*locator)
        except NoSuchElementException:
            continue
        try:
            if el.is_displayed():
                return label
        except WebDriverException:
            continue
    return None


def detect_social_context_retweet(driver):
    """
    偵測「XX 轉發了 / XX Reposted」這種 social context 提示。

    這是轉推的第二種判斷依據，跟網址比對是互補的：網址比對只在
    X 把瀏覽器導向到別的網址時有效；但實測發現更常見的情況是網址
    列完全不變，只是這個位置原地顯示成原推文內容、上面多一行小字
    提示。網址比對抓不到這種，這裡另外用 social context 元素抓。

    selector 是憑常見寫法猜的，沒辦法在這裡實際打開 X 驗證，
    如果之後發現這個判斷沒有生效（redirected_possible_retweet
    數量還是接近 0，但用健檢腳本查出來的重複污染比例仍然很高），
    要回頭用瀏覽器開發者工具核對實際的元素。
    """
    try:
        el = driver.find_element(By.CSS_SELECTOR, '[data-testid="socialContext"]')
    except NoSuchElementException:
        return False
    try:
        if not el.is_displayed():
            return False
    except WebDriverException:
        return False
    text = el.text.lower()
    return any(kw in text for kw in ["repost", "retweet", "轉發", "轉推"])


# ---------------------------------------------------------------------------
# Step E: 抓取單條 tweet
# ---------------------------------------------------------------------------

def scrape_tweet(driver, tweet_id, wait_seconds=8):
    url = f"https://x.com/i/status/{tweet_id}"
    result = {
        "tweet_id": tweet_id,
        "status": "error",
        "text": None,
        "created_at": None,
        "username": None,
        "reply_count": None,
        "retweet_count": None,
        "like_count": None,
        "redirected_to": None,
        "scraped_at": time.time(),
    }
    try:
        driver.get(url)
        time.sleep(2)  # 讓 SPA 完成初次渲染

        block_label = detect_block(driver)
        if block_label:
            # 先就地重試一次，排除單純誤判/載入中的可能。
            time.sleep(5)
            driver.get(url)
            time.sleep(2)
            block_label = detect_block(driver)

        if block_label:
            # 重試後仍然被擋，往外拋，交給 run_forever 重建瀏覽器、
            # 拉長冷卻時間再重試，不自動繞過。
            raise SessionBlockedException(f"blocked: {block_label}")

        # 轉推判斷 (a)：網址是否還是原本要求的 tweet_id。
        url_redirected = tweet_id not in driver.current_url
        # 轉推判斷 (b)：頁面上有沒有 social context 提示。
        social_context_retweet = detect_social_context_retweet(driver)

        if url_redirected or social_context_retweet:
            result["status"] = "redirected_possible_retweet"
            result["redirected_to"] = driver.current_url if url_redirected else None
            return result

        page_text_lower = driver.page_source.lower()

        deleted_markers = [
            "this post is unavailable", "此貼文無法使用",
            "doesn’t exist", "doesn't exist", "帳號不存在",
        ]
        suspended_markers = [
            "suspended account", "帳號已被停權", "account is suspended",
        ]
        protected_markers = [
            "these posts are protected", "protected account", "受保護的推文",
        ]

        if any(m in page_text_lower for m in deleted_markers):
            result["status"] = "deleted_or_unavailable"
            return result
        if any(m in page_text_lower for m in suspended_markers):
            result["status"] = "suspended_account"
            return result
        if any(m in page_text_lower for m in protected_markers):
            result["status"] = "protected_account"
            return result

        wait = WebDriverWait(driver, wait_seconds)
        article = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            )
        )

        try:
            text_el = article.find_element(
                By.CSS_SELECTOR, 'div[data-testid="tweetText"]'
            )
            result["text"] = text_el.text
        except NoSuchElementException:
            pass

        try:
            time_el = article.find_element(By.TAG_NAME, "time")
            result["created_at"] = time_el.get_attribute("datetime")
        except NoSuchElementException:
            pass

        try:
            user_el = article.find_element(
                By.CSS_SELECTOR, 'div[data-testid="User-Name"]'
            )
            result["username"] = user_el.text.split("\n")[-1]
        except NoSuchElementException:
            pass

        for testid, key in [
            ("reply", "reply_count"),
            ("retweet", "retweet_count"),
            ("like", "like_count"),
        ]:
            try:
                el = article.find_element(
                    By.CSS_SELECTOR, f'[data-testid="{testid}"]'
                )
                result[key] = el.text or "0"
            except NoSuchElementException:
                pass

        result["status"] = "ok"

    except SessionBlockedException:
        raise
    except TimeoutException:
        result["status"] = "timeout"
    except WebDriverException as e:
        result["status"] = f"webdriver_error: {str(e)[:100]}"
    except Exception as e:
        result["status"] = f"unexpected_error: {str(e)[:100]}"

    return result


# ---------------------------------------------------------------------------
# Step F: 主流程
# ---------------------------------------------------------------------------

def run_once(tree_dir, progress_file, profile_path, limit=None, label_path=None):
    """
    跑一輪：建立瀏覽器、抓一批待爬的 tweet_id。
    若有提供 label_path，會把 label.txt 裡的來源推文 id 排到最前面
    優先爬，補齊標籤分布不平衡的問題。
    回傳 True 代表「待爬清單已經全部完成」，外層可以停止無窮迴圈；
    回傳 False 代表這輪正常跑完（可能還有下一批要繼續）。
    """
    print("解析 tree 檔案，收集需要爬取的 tweet_id ...")
    all_ids = extract_ids_from_tree_dir(tree_dir)
    print(f"共 {len(all_ids)} 條不重複 tweet_id")

    priority_ids = load_priority_ids(label_path)
    if priority_ids:
        print(f"從 label.txt 讀到 {len(priority_ids)} 個優先（來源推文）id")

    done = load_done_ids(progress_file)
    remaining_total = [tid for tid in all_ids if tid not in done]

    if priority_ids:
        remaining_priority = [tid for tid in remaining_total if tid in priority_ids]
        remaining_rest = [tid for tid in remaining_total if tid not in priority_ids]
        remaining_total = sorted(remaining_priority) + sorted(remaining_rest)
        print(f"其中優先 id 尚未完成的: {len(remaining_priority)} 條")
    else:
        remaining_total = sorted(remaining_total)

    todo = remaining_total[:limit] if limit else remaining_total
    print(
        f"已完成 {len(done)} 條，全部剩餘待爬 {len(remaining_total)} 條，"
        f"本輪處理 {len(todo)} 條"
    )

    if not todo:
        print("待爬清單已全部完成（全部剩餘待爬 = 0，不是這批 limit 跑完就停）。")
        return True

    driver = build_driver(profile_path)
    try:
        with open(progress_file, "a", encoding="utf-8") as out:
            for i, tid in enumerate(todo, 1):
                result = scrape_tweet(driver, tid)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i}/{len(todo)}] {tid} -> {result['status']}")

                time.sleep(random.uniform(3, 6))
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    return False


def run_forever(tree_dir, progress_file, profile_path, limit=None, label_path=None):
    """
    外層無窮迴圈。run_once 若因為 session 掛掉（網路卡死、瀏覽器崩潰、
    webdriver 連線中斷、或偵測到驗證/登入畫面）拋出例外，這裡會攔下
    來、記錄原因、等一段時間再重建瀏覽器繼續。已完成的 tweet_id 靠
    progress_file 續爬，不會重爬也不會漏。全部做完會自動結束；要提前
    停止用 Ctrl+C。
    """
    consecutive_failures = 0
    while True:
        try:
            finished = run_once(tree_dir, progress_file, profile_path, limit, label_path)
            if finished:
                print("全部爬取完成，結束。")
                break
            consecutive_failures = 0
        except KeyboardInterrupt:
            print("\n收到手動中斷（Ctrl+C），結束。")
            break
        except SessionBlockedException as e:
            consecutive_failures += 1
            backoff = min(300 * consecutive_failures, 1800)
            print(
                f"\n[第 {consecutive_failures} 次連續遇到驗證/登入畫面] {str(e)[:200]}\n"
                f"不會自動繞過，{backoff} 秒（約 {backoff // 60} 分鐘）後"
                f"重建瀏覽器並繼續（已完成的部分不會重爬）..."
            )
            time.sleep(backoff)
        except Exception as e:
            consecutive_failures += 1
            backoff = min(30 * consecutive_failures, 300)
            print(
                f"\n[第 {consecutive_failures} 次連續失敗] "
                f"{type(e).__name__}: {str(e)[:200]}\n"
                f"{backoff} 秒後重建瀏覽器並繼續（已完成的部分不會重爬）..."
            )
            time.sleep(backoff)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--tree_dir", required=True)
#     parser.add_argument("--progress_file", default="hydrated_tweets.jsonl")
#     parser.add_argument("--profile_path", required=True)
#     parser.add_argument("--limit", type=int, default=None)
#     parser.add_argument("--label_path", default=None,
#                          help="label.txt 路徑，提供的話會優先爬來源推文")
#     args = parser.parse_args()
#     run_forever(args.tree_dir, args.progress_file, args.profile_path,
#                 args.limit, args.label_path)

if __name__ == "__main__":
    tree_dir = "data/rumdetect2017/twitter15/tree"
    progress_file = "data/rumdetect2017/twitter15/hydrated_tweets.jsonl"
    profile_path = "/home/luhongxuan/snap/firefox/common/.mozilla/firefox/mzAyqv0h.設定檔 1"
    label_path = "data/rumdetect2017/twitter15/label.txt"
    limit = 30
    run_forever(tree_dir, progress_file, profile_path, limit, label_path)