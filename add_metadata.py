import os
import gym, minisat 
from collections import defaultdict
from deepgatesat.utils import build_argparser
from deepgatesat.agents import MiniSATAgent

DEBUG_ROLLOUTS = 10  


def main():
    parser = build_argparser()
    args = parser.parse_args()
    # key is the name of the problem file, value is a list with two values [minisat_steps_no_restarts, minisat_steps_with_restarts]
    results = defaultdict(list)

    for with_restarts in [False, True]:
        env = gym.make(
            "sat-v0",
            args=args,
            problems_paths=args.eval_problems_paths,
            test_mode=True,
            with_restarts=with_restarts,
        )
        agent = MiniSATAgent()
        print(f"Testing agent {agent}... with_restarts is set to {with_restarts}")
        pr = 0
        while env.test_to != 0 or pr == 0:
            observation = env.reset(max_decisions_cap=500)
            done = False
            while not done:
                action = agent.act(observation)
                observation, reward, done, info = env.new_step(action, dummy=True)
            print(
                f'Rollout {pr+1}, steps {env.step_ctr}, num_restarts {info["num_restarts"]}.'
            )
            if env.aig_problem != None:
                results[env.aig_problem].append(env.step_ctr)
            else:
                results[env.curr_problem].append(env.step_ctr)
            pr += 1
            if args.debug and pr >= DEBUG_ROLLOUTS:
                break
        env.close()
    return results, args


from os import path

if __name__ == "__main__":
    results, args = main()
    for pdir in args.eval_problems_paths.split(":"):
        with open(os.path.join(pdir, "METADATA"), "w") as f:
            for el in sorted(results.keys()):
                cur_dir, pname = path.split(el)
                if path.realpath(pdir) == path.realpath(cur_dir):
                    # no restarts/with restarts
                    f.write(f"{pname},{results[el][0]},{results[el][1]}\n")
