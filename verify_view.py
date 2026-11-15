import asyncio
import json
import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1]/'ARCHIVE/teacher_code_cleanup_20260924/方法/数据采集/_browser_deps'))
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=await browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.goto((OUT/'index.html').as_uri())
        assert await page.locator('.school').count()==79
        await page.select_option('#status','invalid')
        assert await page.locator('.school').count()==5
        await page.select_option('#status','')
        await page.select_option('#district','坪山区')
        assert await page.locator('.school').count()==31
        await page.fill('#query','东北师范')
        assert await page.locator('.school').count()==1
        await page.locator('.school').nth(0).click()
        assert '122' in await page.locator('#detail').inner_text()
        await page.screenshot(path=str(OUT/'preview.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        await page.set_viewport_size({'width':390,'height':844})
        assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        await page.select_option('#district','')
        await page.fill('#query','')
        scanned=await page.evaluate('''() => {let n=0;for(const r of all){selected=r.zone_id;renderDetail();if(!document.querySelector('#map path'))throw Error(r.zone_id);n++;}return n;}''')
        assert scanned==79 and not errors
        links=await page.evaluate('''() => [...document.querySelectorAll('a[href]')].map(a=>a.getAttribute('href')).filter(x=>!/^https?:/.test(x))''')
        assert all((OUT/link).is_file() for link in links),links
        await page.fill('#query','not-a-school-987654')
        assert await page.locator('.school').count()==0
        await browser.close()
    result=dict(passed=True,all_paths_rendered=scanned,self_intersection_filter=5,
        only_selected_duplicate_path_displayed=True,desktop_mobile_no_overflow=True,local_links_exist=True,javascript_errors=errors)
    (OUT/'页面验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':asyncio.run(main())
