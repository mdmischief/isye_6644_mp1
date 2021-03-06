import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom
from statistics import median
import argparse
import pathlib

fig_dir = pathlib.Path.cwd() / 'figs'
fig_dir.mkdir(parents=True, exist_ok=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser=argparse.ArgumentParser(
    prog='mp1.py'
    ,description='ISYE 6644 MiniProject1 - Flu.'
)
parser.add_argument(
    "-ns"
    ,"--n_students"
    ,required=False
    ,help="Number of Students"
    ,type=int
    ,dest='n_students'
    ,default=21
)
parser.add_argument(
    "-w"
    ,"--weekend_check"
    ,required=False
    ,help="Account for no school on Weekends"
    ,type=str2bool
    ,dest='weekend_check'
    ,default=True
)
parser.add_argument(
    "-eps"
    ,"--episodes"
    ,required=False
    ,help="Number of episodes to simulate"
    ,type=int
    ,dest='episodes'
    ,default=10000
)
parser.add_argument(
    "-nd"
    ,"--n_days"
    ,required=False
    ,help="Number of days to simulate"
    ,type=int
    ,dest='n_days'
    ,default=63
)
parser.add_argument(
    "-p"
    ,"--probability_infect"
    ,required=False
    ,help="Probabilty of Infection"
    ,type=float
    ,dest='p'
    ,default=0.02
)
parser.add_argument(
    "-ndi"
    ,"--n_days_infectious"
    ,required=False
    ,help="Number of days infectious"
    ,type=int
    ,dest='n_days_infectious'
    ,default=3
)

args=vars(parser.parse_args())
n_students=args['n_students']
weekend_check=args['weekend_check']
eps=args['episodes']
n_days=args['n_days']
p=args['p']
n_days_infectious=args['n_days_infectious']

#
# stu_array: Infected: bool, 0 = No, 1 = Yes
#            Days infected: int, -1,0,1,2,3,4...

stu_array = np.array([[1, 1]] + [[0, 0]]*(n_students - 1))
infections = np.array([1]) # Array to track the number infected per day. Update upon a new infection.

def sim_day(students, infects, day=0, p=0.02):
    """Simulates a day at school and the infections that may occur.
        Given: students: stu_array of type np.array;
                infects: number of students who've been infected of type int
        Returns: students, infects
    """
    if weekend_check and day != 0 and (day+1) % 6 == 0:
        #do nothing
        # print(f'Day = {day}: 6 do nothing')
        pass
    elif weekend_check and day != 0 and (day+1) % 7 == 0:
        #do nothing
        # print(f'Day = {day}: 7 do nothing')
        pass
    else:
        # print(f'Day = {day}: do the thing')
        for stu in range(len(stu_array)):
            if students[stu, 0] == 1 and students[stu,1] in range(1, 4): # Student has caught the flu and is contagious
                for peer in range(len(students)):
                    if students[peer, 0] == 0 and stu != peer: # Peer is susceptible to flu except on weekends
                        rand = np.random.uniform(0,1)
                        if rand <= p: # Peer gets infected
                            students[peer, 0] = 1
                            infects += 1
    return students, infects

def episode(p=0.02):
    """One episode simulates the virus, beginning with Tommy, spreading in a classroom until the virus infects everyone
     or no one is contagious.
        Returns: infections - np array containing number infected by each day"""

    stu_array = np.array([[1, 1]] + [[0, 0]] * (n_students - 1))
    infections = np.zeros(n_days)  # Array to track the number infected by each day. Update upon a new infection.

    # Simulate up to n_days (default: 63) contiguous days at a school
    contagious = True # True if any stu_array[:, 1] in [1,4). Only Tommy (student 0) is contagious on day 1.
    day = 0
    infected = 1
    while contagious and day < n_days: # Each loop sims 1 day
        stu_array, new_infections = sim_day(stu_array, infections[-1], day, p)
        infected += new_infections
        infections[day] = infected

        mask = stu_array[:, 0] == 1
        stu_array[:, 1] = stu_array[:, 1] + mask

        # Check if there's still any contagious students. If not, end sim
        cont_mask = (stu_array[:, 1] >= 1) & (stu_array[:, 1] <= n_days_infectious)
        if not any(cont_mask):
            contagious = False
            infections[day:] = infections[day]
        day += 1

    return infections

def save_table(df, title, name):
    from pandas.plotting import table  # EDIT: see deprecation warnings below
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.set_title(title)
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    table(ax, df)  # where df is your data frame
    plt.savefig(f'{name}.png')

def main():
    # Simulate many episodes
    day_results = np.zeros((eps, n_days)) # Store number infected by each day for each episode
    episodes = 0
    np.random.seed(2**23 - 1)
    while episodes < eps: # Each loop sims 1 episode
        day_results[episodes] = episode(p)
        episodes += 1

    # Calculate and print stats

    # Average number of days epidemics lasted
    epidem_lens = np.argmax(day_results, axis=1) + 4 # Zero-index + 3 (number of days contagious) + 1
    print('Consider a pandemic complete if no infected students remain.')
    print('Mean days pandemic lasted:', epidem_lens.mean())
    print('Median days pandemic lasted:', median(epidem_lens))

    # Average number infected over all episodes
    print('Mean infections:', day_results[:, -1].mean())
    print('Median infections:', median(day_results[:, -1]))

    # Expected value of infected by each day
    expected_values = day_results.mean(axis=0)
    # n_days = 41
    expected_df = pd.DataFrame(list(range(1, n_days+1)), columns=["Day"])
    expected_df['Mean'] = expected_values[:n_days]

    print(expected_df)

    # Part C - Day 2 Expected Value
    n = 20
    r_values = list(range(n + 1)) # Possible numbers infected on day 1
    dist1 = [binom.pmf(r, n, 0.02) for r in r_values] # Day 1
    probs = np.zeros(20)
    for day1 in r_values:
        p3 = 0.02 * (1 + day1)
        dist2 = [binom.pmf(r, n-day1, p3) for r in r_values] # Day 2 given day 1
        #print(dist2)
        for x in range(len(dist2)):
            total = x + day1
            #print('Total', day1, x, dist1[day1], dist2[x])
            if total < 20:
                probs[total] += dist1[day1] * dist2[x]

    probs_row = [[probs[0]], [probs[1]], [probs[2]], [probs[3]], [probs[4]], [probs[5]], [probs[6]], [probs[7]], [probs[8]], [probs[9]], 
                 [probs[10]], [probs[11]], [probs[12]], [probs[13]], [probs[14]], [probs[15]], [probs[16]], [probs[17]], [probs[18]], 
                 [probs[19]]]
    day2_df = pd.DataFrame([[round(b, 5) for b in probs]], columns=[f'{a}' for a in range(1, 21)])
    day2_df.to_csv(fig_dir / 'day2_dist.csv')
    print('Output probability distribution for Part C to "day2_dist.csv"')

    mean2 = (np.arange(20).dot(probs))+1
    print('Expected value of students infected by day 2:', mean2)

    title = f'Histogram of First Two Days of Pandemic (Theoretical)\n {eps:,} Episodes. Mean = {round(mean2, 2)} days'
    plt.bar(x=np.arange(1, 21), height=probs, align='edge', width=1)
    plt.title(title)
    plt.xlim(left=1, right=9)
    plt.xlabel('Infections')
    plt.ylabel('Probability')
    plt.savefig(fig_dir / f'Flu_Pandemic_Fig2_{eps}_weekend_{weekend_check}.png')
    plt.show()

    title = f'Histogram of First Two Days of Pandemic (Empirical)\n {eps:,} Episodes. Mean = {round(expected_values[1], 2)} days'
    plt.hist(day_results[:, 1], bins=range(1, 10))
    plt.title(title)
    plt.xlim(left=1, right=9)
    plt.xlabel('Infections')
    plt.ylabel('Probability')
    plt.savefig(fig_dir / f'Flu_Pandemic_Fig3_{eps}_weekend_{weekend_check}.png')
    plt.show()

    # Part D - Histogram
    title = f'Histogram of Days the Epidemic Lasted\n {eps:,} Episodes. Mean = {round(epidem_lens.mean(), 2)} days, Median = {median(epidem_lens)} days'
    plt.hist(epidem_lens)
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Episodes')
    plt.savefig(fig_dir / f'Flu_Pandemic_Fig1_{eps}_weekend_{weekend_check}.png')
    plt.show()

    title = f'Histogram of Days the Epidemic Lasted\n {eps:,} Episodes. Mean = {round(epidem_lens.mean(), 2)} days, Median = {median(epidem_lens)} days'
    plt.hist(epidem_lens, bins=range(min(epidem_lens), max(epidem_lens)+1, 1))
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Episodes')
    plt.savefig(fig_dir / f'Flu_Pandemic_Fig1_smallbins_{eps}_weekend_{weekend_check}.png')
    plt.show()

    title = f'Line Chart of Mean Cumulative Infections\n {eps:,} Episodes. Mean = {round(expected_df["Mean"].mean(), 2)} infections, Median = {round(median(expected_df["Mean"]), 2)} infections'
    plt.plot(expected_df['Mean'])
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Mean Infections')
    plt.savefig(fig_dir / f'Flu_Pandemic_means_{eps}_weekend_{weekend_check}.png')
    plt.show()
    
    # What p will infect all?
    # Simulate many episodes
    print('Simulating different levels of p...')
    eps2 = 1000 # Number of episodes to simulate
    p_array = np.arange(0.02, 0.16, 0.01)
    day_results2 = np.zeros((eps2, 63, len(p_array))) # Store number infected by each day for each episode for each p
    p_range = np.zeros(shape=(len(p_array), 2))
    for p2 in range(len(p_array)):
        episodes = 0
        np.random.seed(2 ** 23 - 1)
        while episodes < eps2: # Each loop sims 1 episode
            day_results2[episodes, :, p2] = episode(p_array[p2])
            episodes += 1
        p_range[p2, 0] = p_array[p2]
        p_range[p2, 1] = round(len(day_results2[day_results2[:, -1, p2] == 21])/eps2 * 100, 3)
    p_df = pd.DataFrame(p_range, columns=['p', '% Fully Infected'])
    p_df.to_csv(fig_dir / 'p_range.csv')
    print('Sim complete. Output results to "p_range.csv"')

if __name__ == "__main__":
    main()
