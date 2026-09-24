"""
TCG-AI 自动化压力测试与稳定性自检脚本 (连续平衡收敛检验)
【核心定义】:
每跑完一整趟调优流水线并最终判定“达成平衡收敛”，算作 1 次成功测试。
压力测试的目标是检验调优调度与算法系统能否经受连续考验，达成【连续成功 N 次】(例如连着成功 10 次)！
每轮测试完成后，自动清理上一轮生成的 PPO 模型权重，模拟全新环境重新探索自平衡。
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import shutil
import time
import json
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_stress_test(
    target_consecutive=10,
    episodes=300,
    max_outer=6,
    target_balance=5.0,
    target_pairwise=8.0,
    dry_run=False,
    reset_cards=False,
    max_total_attempts=20,
    skip_final_audit=True
):
    print("=" * 75)
    print(f"  🏆 TCG-AI 连续平衡收敛压力测试启动")
    print(f"  ★ 核心目标: 连续成功达成平衡收敛 【{target_consecutive} 次】 (中途不失衡、不报错、不卡死)")
    print(f"  ★ 对局配置: 单轮混战规模 {episodes if not dry_run else 60} 局 | 综合容差 ±{target_balance:.1f}% | 两两容差 ±{target_pairwise:.1f}%")
    print(f"  ★ 运行模式: {'[极速演练模式]' if dry_run else '[实机对决评测]'} | 卡池策略: {'[每次重置基准卡池]' if reset_cards else '[连续继承演化(检验卡池抗压性)]'}")
    print(f"  ★ 容错上限: 最多尝试 {max_total_attempts} 轮 (若连胜中断自动重新累计，直至达成连续 {target_consecutive} 次)")
    print("=" * 75)

    cards_path = os.path.join(SCRIPT_DIR, "cards_config.json")
    baseline_backup = os.path.join(SCRIPT_DIR, "cards_config_stress_backup.json")

    # 备份初始基准卡池
    if os.path.exists(cards_path):
        shutil.copy2(cards_path, baseline_backup)
        print(f"[*] 已建立基准卡池快照: {baseline_backup}")

    history_results = []
    consecutive_streak = 0
    max_streak_reached = 0
    total_attempts = 0
    total_successes = 0

    orchestrator_path = os.path.join(SCRIPT_DIR, "pipeline_orchestrator.py")
    test_start_time = time.time()

    try:
        while consecutive_streak < target_consecutive and total_attempts < max_total_attempts:
            total_attempts += 1
            print("\n" + "=" * 75)
            print(f"【压力测试 尝试第 {total_attempts} 轮】 当前连胜进度: [ {consecutive_streak} / {target_consecutive} ] 次")
            print("=" * 75)

            # 1. 彻底清理上一轮生成的 PPO 模型权重，模拟全新冷启动智能体探索
            weight_file = os.path.join(SCRIPT_DIR, "card_ppo_model_brawl.pth")
            if os.path.exists(weight_file):
                try:
                    os.remove(weight_file)
                    print(f"  [清理] 已重置 PPO 权重: {os.path.basename(weight_file)} (确保从零全新学习)")
                except Exception as e:
                    print(f"  [警告] 删除旧权重失败: {e}")

            # 2. 清理 Tabu 禁忌列表
            tabu_file = os.path.join(SCRIPT_DIR, "balancer_tabu_list.json")
            if os.path.exists(tabu_file):
                try:
                    os.remove(tabu_file)
                    print(f"  [清理] 已重置 Tabu 记忆: {os.path.basename(tabu_file)}")
                except Exception:
                    pass

            # 3. 恢复基准卡池（如果启用了每次重置模式）
            if reset_cards and os.path.exists(baseline_backup):
                shutil.copy2(baseline_backup, cards_path)
                print("  [恢复] 已重置卡池为初始基准数据")

            # 4. 构建执行命令
            actual_episodes = 60 if dry_run else episodes
            cmd = [
                sys.executable, orchestrator_path,
                "--skip-print",
                "--skip-art",
                "--episodes", str(actual_episodes),
                "--max-outer-iterations", str(max_outer),
                "--target-balance", str(target_balance),
                "--target-pairwise-balance", str(target_pairwise)
            ]
            if dry_run:
                cmd.append("--dry-run")
            if skip_final_audit:
                cmd.append("--skip-final-audit")

            log_file = os.path.join(SCRIPT_DIR, f"stress_test_log_run_{total_attempts}.txt")
            print(f"  [启动] 执行指令: {' '.join(cmd)}")
            print(f"  [监控] 实时输出已记录至: {os.path.basename(log_file)}")
            print("-" * 75)

            start_t = time.time()
            last_key_info = "流水线执行中..."

            # 在 Windows 下设置后台运行优先级 (BELOW_NORMAL_PRIORITY_CLASS = 0x00004000)
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = 0x00004000

            with open(log_file, "w", encoding="utf-8") as f_log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creation_flags
                )

                for line in proc.stdout:
                    f_log.write(line)
                    f_log.flush()
                    line_s = line.strip()

                    # 智能筛选关键状态在控制台展示，让用户实时掌握平衡演化进展
                    if any(k in line_s for k in [
                        "[Episode", "【轮次", "[判定]", "达成平衡收敛", "【对战遥测统计结果】",
                        "【阶段四】", "【阶段五】", "微调记录", "[严重失衡熔断]",
                        "DeepSeek 大模型", "[数值调整]", "[安全兜底]", "已完成卡池数值调优",
                        "达到最大迭代轮次", "★"
                    ]):
                        print(f"  {line_s}")
                        if "达成平衡收敛" in line_s or "胜率达成平衡" in line_s or "退出迭代循环" in line_s:
                            last_key_info = "达成平衡收敛条件"
                        elif "已完成卡池数值调优" in line_s:
                            last_key_info = "已调优卡池数值"
                        elif "对战遥测统计结果" in line_s:
                            last_key_info = "实机遥测审计中"

                proc.wait()
                end_t = time.time()
                duration = end_t - start_t

            is_success = (proc.returncode == 0)

            if is_success:
                total_successes += 1
                consecutive_streak += 1
                if consecutive_streak > max_streak_reached:
                    max_streak_reached = consecutive_streak

                print("-" * 75)
                print(f"🎉【第 {total_attempts} 轮结果】: ✅ 平衡收敛成功！耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
                print(f"🔥【连胜计数器】: 当前已连续成功 [ {consecutive_streak} / {target_consecutive} ] 次！")
                if consecutive_streak >= target_consecutive:
                    print(f"🏆【目标达成】: 恭喜！已圆满达成连续 {target_consecutive} 次平衡收敛的压力测试目标！")
            else:
                prev_streak = consecutive_streak
                consecutive_streak = 0
                print("-" * 75)
                print(f"❌【第 {total_attempts} 轮结果】: 失败/未收敛 (退出码: {proc.returncode}) | 耗时: {duration:.1f} 秒")
                print(f"⚠️【连胜中断】: 连续成功计数在第 {prev_streak} 次中断，连胜计数重置为 0！")
                print(f"   详细排查日志已保存至: {log_file}")

            history_results.append({
                "attempt": total_attempts,
                "success": is_success,
                "streak_after": consecutive_streak,
                "duration_sec": duration,
                "returncode": proc.returncode,
                "last_info": last_key_info
            })

    except KeyboardInterrupt:
        print("\n\n[中断] 用户手动终止了压力测试！")

    finally:
        # 清理基准备份
        if os.path.exists(baseline_backup):
            try:
                os.remove(baseline_backup)
            except Exception:
                pass

    total_test_duration = time.time() - test_start_time

    # 输出高压测试终局大面板
    print("\n" + "=" * 75)
    print("           🏆 TCG-AI 自动化压力测试总结报告 (连续平衡收敛检验)")
    print("=" * 75)
    print(f" 目标连续成功: {target_consecutive} 次")
    print(f" 实际最高连胜: {max_streak_reached} 次 ({'✅ 达标' if max_streak_reached >= target_consecutive else '⚠️ 未达标'})")
    print(f" 总测试尝试轮数: {total_attempts} 轮")
    print(f" 累计成功收敛数: {total_successes} 轮 (总体成功率: {total_successes / max(1, total_attempts) * 100:.1f}%)")
    print(f" 累计测试总耗时: {total_test_duration:.1f} 秒 ({total_test_duration/60:.1f} 分钟)")
    print("\n[逐轮战报详情]")
    for r in history_results:
        st = "✅ 成功收敛" if r["success"] else f"❌ 失败/中断 (code={r['returncode']})"
        print(f"  * 尝试第 {r['attempt']:02d} 轮: {st} | 耗时: {r['duration_sec']:5.1f}s | 连胜: {r['streak_after']}/{target_consecutive} | 状态: {r['last_info']}")
    print("=" * 75)

    if max_streak_reached >= target_consecutive:
        print("【结论】: 🎉 压力测试完美通过！系统在连续高频探索与调优中表现出极佳的收敛稳定性与健壮性！")
    else:
        print("【结论】: ⚠️ 未能达成连续目标次数，请结合上述对应轮次的 stress_test_log_run_X.txt 排查偶发失衡原因。")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("   TCG-AI 自动化压力测试与稳定性自检 (连续平衡收敛检验)")
        print("=" * 70)
        print("【压力测试定义】:")
        print("  跑完一整趟调优流水线并最终判定“达成平衡收敛”，算作 1 次成功测试。")
        print("  目标是检验系统能否经受连续高频考验，达成【连续成功 N 次】(连着成功十次)！")
        print("  每轮跑完自动清空旧 PPO 权重，确保每一轮都是从零全新验证。")
        print("\n请选择压力测试方案：")
        print(" [1] 标准连续收敛压测 (每轮 300 局，连胜 10 次达成平衡，兼顾真实统计与速度，约 12~15 分钟) [推荐]")
        print(" [2] 极速演练冒烟测试 (每轮几十局快速演练，连胜 10 次，用于秒级跑通调度与死锁排查)")
        print(" [3] 深度高精收敛压测 (每轮 800 局，连胜 10 次，严苛实机极限平衡抗压检验)")
        print(" [4] 自定义配置 (自定义连续成功目标次数、单轮局数、容差与卡池策略)")
        print()

        try:
            choice = input("请输入选项编号 (直接回车默认 [1]): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已取消。")
            sys.exit(0)

        if choice == "2":
            run_stress_test(target_consecutive=10, dry_run=True, target_pairwise=15.0)
        elif choice == "3":
            run_stress_test(target_consecutive=10, episodes=800, max_outer=6, target_pairwise=8.0, dry_run=False)
        elif choice == "4":
            try:
                n_str = input("请输入目标连续成功次数 (默认 10): ").strip()
                t_consecutive = int(n_str) if n_str else 10

                ep_str = input("请输入每轮对局规模 (默认 300): ").strip()
                t_episodes = int(ep_str) if ep_str else 300

                pair_str = input("请输入两两对抗容差 %% (默认 8.0): ").strip()
                t_pairwise = float(pair_str) if pair_str else 8.0

                rst_str = input("卡池策略 [1: 连续继承演化(推荐) | 2: 每次重置基准] (默认 1): ").strip()
                t_reset = (rst_str == "2")

                run_stress_test(
                    target_consecutive=t_consecutive,
                    episodes=t_episodes,
                    max_outer=6,
                    target_pairwise=t_pairwise,
                    dry_run=False,
                    reset_cards=t_reset
                )
            except Exception as e:
                print(f"输入有误: {e}，将采用默认标准模式启动。")
                run_stress_test(target_consecutive=10, episodes=300, max_outer=6, target_pairwise=8.0, dry_run=False)
        else:
            # 默认选项 1: 标准连续收敛压测 (连胜 10 次)
            run_stress_test(target_consecutive=10, episodes=300, max_outer=6, target_pairwise=8.0, dry_run=False)
    else:
        parser = argparse.ArgumentParser(description="TCG-AI 自动化连续平衡收敛稳定性压力测试工具")
        parser.add_argument("-n", "--consecutive", type=int, default=10, help="目标连续成功次数 (默认 10 次，连胜 10 轮达成平衡)")
        parser.add_argument("--episodes", type=int, default=300, help="每轮对局规模 (默认 300 局)")
        parser.add_argument("--max-outer", type=int, default=6, help="单次测试最大数值微调上限 (默认 6 轮)")
        parser.add_argument("--target-balance", type=float, default=5.0, help="综合胜率偏离容差 (默认 5.0%%)")
        parser.add_argument("--target-pairwise", type=float, default=8.0, help="两两对抗容差 (默认 8.0%%)")
        parser.add_argument("--dry-run", action="store_true", help="极速演练模式 (每轮仅几十局，用于秒级跑通流程)")
        parser.add_argument("--reset-cards", action="store_true", help="每次重置卡池为初始基准 (默认继承演化)")
        parser.add_argument("--max-attempts", type=int, default=20, help="最大允许尝试总轮数 (默认 20 轮)")
        args = parser.parse_args()

        run_stress_test(
            target_consecutive=args.consecutive,
            episodes=args.episodes,
            max_outer=args.max_outer,
            target_balance=args.target_balance,
            target_pairwise=args.target_pairwise,
            dry_run=args.dry_run,
            reset_cards=args.reset_cards,
            max_total_attempts=args.max_attempts
        )
