import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_students = 21

#
# stu_array: Infected: bool, 0 = No, 1 = Yes
#            Days infected: int, -1,0,1,2,3,4...

stu_array = np.array([[1, 1]] + [[0, 0]]*(n_students - 1))
infections = np.array([1]) # Array to track the number infected per day. Update upon a new infection.

def sim_day(students, infects, day):
    """Simulates a day at school and the infections that may occur.
        Given: students: stu_array of type np.array;
                infects: number of students who've been infected of type int
        Returns: students, infects
    """
    p = 0.02

    if day != 0 and (day+1) % 6 == 0:
        #do nothing
        # print(f'Day = {day}: 6 do nothing')
        pass
    elif day != 0 and (day+1) % 7 == 0:
        #do nothing
        # print(f'Day = {day}: 7 do nothing')
        pass
    else:
        # print(f'Day = {day}: do the thing')
        for stu in range(len(stu_array)):
            if students[stu, 0] == 1 and students[stu,1] in range(1, 4): # Student has caught the flu and is contagious
                for peer in range(len(students)):
                    if students[peer, 0] == 0 and stu != peer: # Peer is suscetpible to flu except on weekends
                        rand = np.random.uniform(0,1)
                        if rand <= p: # Peer gets infected
                            students[peer, 0] = 1
                            infects += 1
    return students, infects

def episode():
    """One episode simulates the virus, beginning with Tommy, spreading in a classroom until the virus infects everyone
     or no one is contagious.
        Returns: infections - np array containing number infected by each day"""

    stu_array = np.array([[1, 1]] + [[0, 0]] * (n_students - 1))
    infections = np.zeros(100)  # Array to track the number infected by each day. Update upon a new infection.

    # Simulate up to 63 contiguous days at a school
    contagious = True # True if any stu_array[:, 1] in [1,4). Only Tommy (student 0) is contagious on day 1.
    day = 0
    infected = 1
    while contagious: # Each loop sims 1 day
        stu_array, new_infections = sim_day(stu_array, infections[-1], day)
        infected += new_infections
        infections[day] = infected

        mask = stu_array[:, 0] == 1
        stu_array[:, 1] = stu_array[:, 1] + mask

        # Check if there's still any contagious students. If not, end sim
        cont_mask = (stu_array[:, 1] >= 1) & (stu_array[:, 1] <= 3)
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

# Simulate many episodes
eps = 10 ** 4 # Number of episodes to simulate
day_results = np.zeros((eps, 100)) # Store number infected by each day for each episode
episodes = 0
np.random.seed(2**23 - 1)
while episodes < eps: # Each loop sims 1 episode
    day_results[episodes] = episode()
    episodes += 1

# Calculate and print stats

# Average number of days epidemics lasted
epidem_lens = np.argmax(day_results, axis=1) + 4 # Zero-index + 3 (number of days contagious) + 1
print('Consider a pandemic complete if no infected students remain.')
print('Mean days pandemic lasted:', epidem_lens.mean())
from statistics import median
print('Median days pandemic lasted:', median(epidem_lens))

# Average number infected over all episodes
print('Mean infections:', day_results[:, -1].mean())
print('Median infections:', median(day_results[:, -1]))

# Expected value of infected by each day
expected_values = day_results.mean(axis=0)
n_days = 100
expected_df = pd.DataFrame(range(1, n_days+1), columns=["Day"])
expected_df['Mean'] = expected_values[:n_days]
#save_table(expected_df, f'Expected Number of Infected Students Per Day\nMonte Carlo with {eps} Simulations', 'day_means')
from IPython.display import display
display(expected_df)

#Day 2 Expected Value

# Does this include Timmy the PatientZero?
from scipy.stats import binom

n = 20
p = 0.02


r_values = list(range(n + 1))
dist1 = [binom.pmf(r, n, p) for r in r_values] # Day 1
probs = np.zeros(20)
for day1 in range(20):
    dist2 = [binom.pmf(r, n-day1, p) for r in r_values] # Day 2 given day 1
    for x in range(len(dist2)):
        total = x + day1
        if total < 20:
            probs[total] += dist1[day1] * dist2[x]

mean2 = np.arange(20).dot(probs)
print('Expected value of students infected by day 2:', mean2)

from bokeh.io import show, output_file
from bokeh.plotting import figure

# title = f'Histogram of Days the Epidemic Lasted\n {eps} Episodes. Mean = {round(epidem_lens.mean(), 2)} days, Median = {median(epidem_lens)} days'
# plt.hist(epidem_lens)
# plt.title(title)
# plt.xlabel('Days')
# plt.ylabel('Simulations')
# plt.savefig('Flu_Pandemic_Fig1.png')
# plt.show()
# print(len(epidem_lens['days']))
hist, edges = np.histogram(epidem_lens, density=True, bins=n_days)
p = figure()
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")

# output_file("hist.html")
show(p)