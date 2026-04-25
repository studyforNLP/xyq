from config import DEFAULT_TARGET_FILE
from output import print_solution, pretty_print
from parameters import FINAL_SOLUTION_PARAMS, PARAMETER_DESCRIPTIONS, TRAINING_PARAMS
from parser import parse_custom_file
from qlearning import init_q_table
from state_space import ACTIONS, generate_state_space
from training import build_final_solution, sample_training_loop


def _max_skill_level(people_num: int | None) -> int:
    """Return the number of SBI levels minus one."""
    if people_num is None:
        return 4
    if people_num <= 5:
        return people_num - 1
    return 4


def _print_parameter_definitions() -> None:
    print("\n参数定义:")
    for name, value in {**TRAINING_PARAMS, **FINAL_SOLUTION_PARAMS}.items():
        print(f"  {name} = {value}  # {PARAMETER_DESCRIPTIONS[name]}")


def main() -> None:
    """Run the complete Rollout + Q-learning scheduling workflow."""
    target_file = DEFAULT_TARGET_FILE
    if not target_file.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {target_file}\n"
            "请确认路径是否正确，或修改 config.py 中的 DEFAULT_TARGET_FILE。"
        )

    print("正在解析数据文件...")
    parsed = parse_custom_file(target_file)
    print("数据解析完成！")
    pretty_print(parsed)

    max_skill_level = _max_skill_level(parsed.people_num)
    print(f"\n人员总数: {parsed.people_num}")
    print(f"技能瓶颈指数档数: {max_skill_level + 1} (索引 0-{max_skill_level})")
    print("状态特征: (NLF, SBI, MUR, RUR, CRT, ITN)")
    dynamic_state_space = generate_state_space(parsed.people_num)

    _print_parameter_definitions()

    print("\n正在初始化 Q 表...")
    Q_table = init_q_table(dynamic_state_space, ACTIONS)
    print(f"Q 表初始化完成，状态空间大小: {len(dynamic_state_space)}")

    print("\n开始 Rollout + Q-learning 训练...")
    Q_table = sample_training_loop(
        Q_table=Q_table,
        parsed=parsed,
        max_skill_level=max_skill_level,
        **TRAINING_PARAMS,
    )
    print("训练完成！")

    print("\n正在构建最终调度方案...")
    solution, total_penalty = build_final_solution(
        Q_table,
        parsed,
        max_skill_level,
        **FINAL_SOLUTION_PARAMS,
    )
    print_solution(solution, total_penalty, parsed)

    print("\n" + "=" * 80)
    print("算法执行完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
