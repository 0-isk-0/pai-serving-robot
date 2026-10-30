"""Build the report PDF from report/template.html + results/*.json (numbers are never copied by hand).

usage: python report/build_report.py   (run from the repository root; needs Google Chrome for the PDF)
"""
import json
import os
import string
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PDF = os.path.join(ROOT, "report", "説明資料_PAI最終課題_isk091223.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
METHODS = [("go_to_goal", "直進制御（センサなし）"),
           ("potential_field", "ポテンシャル法（古典・調整済み）"),
           ("potential_field_shield", "ポテンシャル法＋安全層"),
           ("noshield", "PPO単体（安全層なし）"),
           ("nocup", "PPO＋安全層／コップなしで学習"),
           ("nodr", "PPO＋安全層／物理ランダム化なし"),
           ("main", "<b>提案：PPO＋安全層</b>")]


def pct(m, sd=None):
    return f"{100 * m:.1f}%" + (f" <span class='sd'>±{100 * sd:.1f}</span>" if sd else "")


def table(summary, scen):
    rows = ["<tr><th>手法</th><th>成功</th><th>家具・壁に衝突</th><th>人と接触</th><th>こぼした</th><th>時間切れ</th>"
            "<th>成功時の所要時間</th></tr>"]
    for key, label in METHODS:
        v = summary[scen].get(key)
        if not v:
            continue
        multi = v["n_seeds"] > 1
        cells = [pct(v["success"][0], v["success"][1] if multi else None)]
        cells.append(pct(v["collision"][0]))                     # outcomes are exclusive: furniture/walls only
        cells += [pct(v[k][0]) for k in ("hit_person", "spill", "timeout")]
        cells.append(f"{v['mean_time_s']:.1f} 秒" if v["mean_time_s"] else "—")
        cls = " class='hl'" if key == "main" else ""
        rows.append(f"<tr{cls}><td>{label}{'（3シード平均±標準偏差）' if multi else ''}</td>" +
                    "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table class='res'>" + "".join(rows) + "</table>"


def readme_table(summary):
    out = []
    for scen, title in (("standard", "標準条件"), ("stress", "厳しい条件（常に4人＋物理パラメータの幅1.6倍）")):
        out.append(f"\n**{title}**\n\n| 手法 | 成功 | 衝突・人との接触 | こぼした | 時間切れ |\n|---|---|---|---|---|")
        for key, label in METHODS:
            v = summary[scen].get(key)
            if not v:
                continue
            sd = f" ±{100 * v['success'][1]:.1f}" if v["n_seeds"] > 1 else ""
            label = label.replace("<b>", "**").replace("</b>", "**") + ("（3シード）" if v["n_seeds"] > 1 else "")
            out.append(f"| {label} | {100 * v['success'][0]:.1f}%{sd} | {100 * (v['collision'][0] + v['hit_person'][0]):.1f}% | "
                       f"{100 * v['spill'][0]:.1f}% | {100 * v['timeout'][0]:.1f}% |")
    return "\n".join(out) + "\n\n"


def main():
    s = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    a = json.load(open(os.path.join(ROOT, "results", "analysis.json")))
    ev = json.load(open(os.path.join(ROOT, "results", "eval.json")))
    m, ms = s["standard"]["main"], s["stress"]["main"]
    pf, gg = s["standard"]["potential_field"], s["standard"]["go_to_goal"]
    vals = dict(
        table_standard=table(s, "standard"), table_stress=table(s, "stress"),
        main_succ=pct(m["success"][0]), main_succ_sd=f"{100 * m['success'][1]:.1f}",
        main_stress=pct(ms["success"][0]), main_spill=pct(m["spill"][0]), main_coll=pct(m["collision"][0] + m["hit_person"][0]),
        main_time=f"{m['mean_time_s']:.1f}", pf_succ=pct(pf["success"][0]), pf_timeout=pct(pf["timeout"][0]),
        pf_time=f"{pf['mean_time_s']:.1f}", gg_succ=pct(gg["success"][0]), gg_coll=pct(gg["collision"][0] + gg["hit_person"][0]),
        pf_stress=pct(s["stress"]["potential_field"]["success"][0]),
        noshield_coll=pct(s["standard"]["noshield"]["collision"][0] + s["standard"]["noshield"]["hit_person"][0]),
        noshield_succ=pct(s["standard"]["noshield"]["success"][0]),
        nocup_spill=pct(s["standard"]["nocup"]["spill"][0]), nocup_succ=pct(s["standard"]["nocup"]["success"][0]),
        nodr_stress=pct(s["stress"]["nodr"]["success"][0]), nodr_std=pct(s["standard"]["nodr"]["success"][0]),
        main_s0_stress=pct(ev["stress"]["main_s0"]["summary"]["success"]),
        latency=f"{a['policy_latency_ms']:.2f}", shield_rate=f"{100 * a['shield_active_rate']:.0f}",
        n_fail=len(a["failures"]), n_ana=a["episodes"],
    )
    sbp = a["spill_by_people"]
    few = sum(v["spills"] for k, v in sbp.items() if int(k) <= 1)
    many = sum(v["spills"] for k, v in sbp.items() if int(k) >= 2)
    n_spill = sum(1 for f in a["failures"] if f["kind"] == "spill")
    n_spill_shield = sum(1 for f in a["failures"] if f["kind"] == "spill" and f["shield_in_last_0_5s"] > 0)
    ns, nc, nd = s["standard"]["noshield"], s["standard"]["nocup"], s["stress"]["nodr"]
    st = [ev["stress"][f"main_s{i}"]["summary"]["success"] for i in range(3)]
    seed_range = f"{100 * min(st):.1f}〜{100 * max(st):.1f}%"
    vals["discussion"] = f"""
<ul>
<li><b>古典手法との比較：</b>ポテンシャル法は一度もぶつからず、こぼさなかったが、テーブル脇のゴールに近づくと反発力と引力がつり合って立ち往生し、
{pct(pf["timeout"][0])} が時間切れになった。提案手法は {vals["main_succ"]} で届け、成功時の所要時間も {m["mean_time_s"]:.1f} 秒（ポテンシャル法 {pf["mean_time_s"]:.1f} 秒）と短い。
一方で、到着直前1秒の最大加速度は提案手法 {m["arrival_acc"]:.1f} m/s²、ポテンシャル法 {pf["arrival_acc"]:.1f} m/s² で、提案手法は「速く走って最後にしっかり減速する」運び方をしている。
コップは倒れていないが余裕は小さく、乗り心地の面では古典手法の方が穏やかである。</li>
<li><b>安全層の効果（PPO単体との比較）：</b>安全層なしで学習・実行した方策は家具・壁への衝突が {pct(ns["collision"][0])}、人との接触が {pct(ns["hit_person"][0])}（成功 {pct(ns["success"][0])}）。
安全層を付けた提案手法では、それぞれ {pct(m["collision"][0])}、{pct(m["hit_person"][0])} で、学習だけでは狭い場所での減速を十分に身につけられないことを、物理に基づく安全層が補っている。
安全層が実際に指令を修正したのは全ステップの {vals["shield_rate"]}% で、運転の大部分は学習した方策が行っている。
なお、ポテンシャル法に同じ安全層を付けても結果は変わらなかった（もともと遅く、安全層が働く場面がない）。</li>
<li><b>コップを意識した学習の効果：</b>コップを載せずに学習した方策は、同じ安全層があってもコップを載せると {pct(nc["spill"][0])} でこぼした（提案手法 {vals["main_spill"]}）。
「ぶつからない」ことと「物を運ぶ品質」は別の能力であり、後者は学習時に物理的に体験させる必要がある。</li>
<li><b>物理ランダム化の効果は確認できなかった：</b>物理パラメータの幅を学習時の1.6倍に広げた厳しい条件で、ランダム化なしで学習した方策の成功は {pct(nd["success"][0])}、
提案手法は {vals["main_stress"]}（3シードの範囲 {seed_range}）だった。差は乱数シードによるばらつきと同程度で、今回の設定ではランダム化の効果は示せていない。
本環境の速度指令は速度フィードバック付きのモーターで実行されるため、質量や摩擦の変化が動きに表れにくいことが一因と考えている（未検証の推測）。
効果を確かめるには、ランダム化なしのモデルも複数シード学習し、遅延やモーターの飽和など、より実機に近い外乱を加えて比べる必要がある。</li>
<li><b>残る失敗の中身：</b>提案手法（seed 0）の失敗は主に「こぼす」で、{n_spill} 件のこぼしのうち{"すべて" if many == n_spill else f" {many} 件"}が2人以上が歩いている場面だった（0〜1人では {few} 件）。
{n_spill} 件中 {n_spill_shield} 件では直前 0.5 秒以内に安全層が減速しており、人を避ける急な操作と安全層の急ブレーキの両方が原因になっている。</li>
</ul>"""
    vals["limitations"] = """
<ul>
<li>シミュレーションのみで、実機では確かめていない。実機では LiDAR の遅延、床の段差、コップの中の液体の揺れ（本研究は剛体で近似）への対応が必要。</li>
<li>人は決まった経路を歩き、ロボットが正面にいると止まる単純なモデルである。人の流れの予測や、人がロボットを避ける動きは扱っていない。</li>
<li>到着直前の減速が強めで、コップが倒れない範囲ではあるが余裕は小さい。到着前の減速を早める報酬設計（加速度の罰をゴール付近で強めるなど）が次の改善点である。</li>
<li>今後：ROS 2 ノードとして方策と安全層を実装し、実機の移動ロボット（例：TurtleBot 系）にトレーを載せて検証したい。
また、注文の自然言語（「窓際のテーブルに水を2つ」）から配膳順を決める上位の計画に言語モデルを使い、本方策を下位の移動に使う構成も考えられる。</li>
</ul>"""
    tpl = string.Template(open(os.path.join(ROOT, "report", "template.html"), encoding="utf-8").read())
    html_path = os.path.join(ROOT, "report", "report.html")
    open(html_path, "w", encoding="utf-8").write(tpl.substitute(vals))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", "file://" + html_path], check=True, capture_output=True)
    print("wrote", OUT_PDF)
    readme = os.path.join(ROOT, "README.md")
    txt = open(readme, encoding="utf-8").read()
    head, rest = txt.split("<!-- results -->", 1)
    _, tail = rest.split("<!-- /results -->", 1)
    open(readme, "w", encoding="utf-8").write(head + "<!-- results -->\n" + readme_table(s) + "<!-- /results -->" + tail)
    print("updated README results table")


if __name__ == "__main__":
    main()
