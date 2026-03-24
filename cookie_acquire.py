import asyncio
import argparse
import json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='获取浏览器状态并保存到指定标签的cookie文件中')
    parser.add_argument('--tag', type=str,  default='xhs', help='测试用例标签，例如 ')
    parser.add_argument('--start-url', type=str, default='https://www.xiaohongshu.com/')
    args = parser.parse_args()

    # 准备保存路径
    storage_file = Path(f"cookie_{args.tag}.json")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # 打开新页面并进行一些交互
        page = await context.new_page()
        await page.goto(args.start_url)

        # 等待用户确认导出 storage_state
        input(f"请在浏览器中完成必要操作，然后按回车键导出 cookie 到 {storage_file}: ")

        dbs = await page.evaluate("""
        async () => {
            if (!indexedDB.databases) {
                return {error: 'indexedDB.databases() not supported in this browser.'};
            }
            const dbInfos = await indexedDB.databases();
            const result = {};
            for (const dbInfo of dbInfos) {
                const dbName = dbInfo.name;
                const dbRequest = indexedDB.open(dbName);
                const db = await new Promise((resolve, reject) => {
                    dbRequest.onsuccess = e => resolve(e.target.result);
                    dbRequest.onerror = e => reject(e);
                });
                const storeNames = Array.from(db.objectStoreNames);
                result[dbName] = {};
                for (const storeName of storeNames) {
                    result[dbName][storeName] = [];
                    await new Promise((resolve, reject) => {
                        const tx = db.transaction([storeName], 'readonly');
                        const store = tx.objectStore(storeName);
                        const cursorReq = store.openCursor();
                        cursorReq.onsuccess = function(e) {
                            const cursor = e.target.result;
                            if (cursor) {
                                result[dbName][storeName].push({
                                    key: cursor.key,
                                    value: cursor.value
                                });
                                cursor.continue();
                            } else {
                                resolve();
                            }
                        };
                        cursorReq.onerror = function(e) {
                            reject(e);
                        };
                    });
                }
                db.close();
            }
            return result;
        }
        """)

        # 导出 storage_state
        await context.storage_state(path=str(storage_file))
        print(f'Cookie 已保存到: {storage_file}')

        # 读取原有 JSON 文件
        with open(storage_file, "r", encoding="utf-8") as f:
            storage_data = json.load(f)

        # 追加 dbs 数据
        storage_data["indexeddb"] = dbs

        # 写回文件
        with open(storage_file, "w", encoding="utf-8") as f:
            json.dump(storage_data, f, ensure_ascii=False)

        print(f'Cookie 和 IndexedDB 已保存到: {storage_file}')

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())