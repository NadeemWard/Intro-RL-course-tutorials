import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import copy
from matplotlib import cm
import numpy as np

def plot_values(state_val):

    if type(state_val) == dict:
        state_val = change_rep(state_val)

    # without usable ace
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')

    # specify range of display for player and dealer
    player_range = np.arange(11, 22)
    dealer_range = np.arange(1, 11)

    # create a meshgrid of each point in the player_range X dealer_range
    X, Y = np.meshgrid(dealer_range, player_range)

    # get the Z-coords
    Z = state_val[11:22, 1:11, 0].reshape(X.shape)  # no usable ace

    # plot x-coords, y-coords, z, coords
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1,
                    rstride=1, cstride=1)
    ax.set_title("Without Ace")
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("State Value")
    plt.show()

    # With usable ace
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(dealer_range, player_range)

    Z = state_val[11:22, 1:11, 1].reshape(X.shape)  # usable ace

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1,
                    rstride=1, cstride=1)
    ax.set_title("With Ace")
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("State Value")
    plt.show()

def change_policy_rep(policy):

    '''
    Change policy from a dict to a numpy array
    :param policy: a dictionary of the form: (state): [prob_a1, prob_a2], ...
    :return: numpy array
    '''

    player_state_space = 32
    dealer_state_space = 11
    ace_state_space = 2
    action_space = 2

    new_policy = np.zeros((player_state_space, dealer_state_space, ace_state_space, action_space))

    for state, value in policy.items():
        player_score, dealer_card, ace = state
        new_policy[player_score][dealer_card][int(ace)][0] = value[0]
        new_policy[player_score][dealer_card][int(ace)][1] = value[1]

    return new_policy

def change_policy_shape(policy, ace):


    if type(policy) == dict:

        policy = copy.deepcopy(policy)
        policy = change_policy_rep(policy)

    player_range = np.arange(11, 22)  # consider from 11 to 21
    dealer_range = np.arange(1, 11)  # consider from 1 to 10

    p = np.zeros((11, 10))

    for i in player_range:
        for j in dealer_range:

            p[i - 11][j - 1] = np.argmax(policy[i][j][ace])

    return p

def change_rep(v):
    '''
    Changes state values from dictionaries to np arrays
    '''

    player_state_space = 32
    dealer_state_space = 11
    ace_state_space = 2

    new_v = np.zeros( (player_state_space, dealer_state_space, ace_state_space) )

    for state, value in v.items():

        player_score, dealer_card, ace = state
        new_v[player_score][dealer_card][int(ace)] = value

    return new_v

def change_rep_q(q):

    '''
    Change action-values from dictionaries to np arrays
    :param q:
    :return: np array of q
    '''

    player_state_space = 32
    dealer_state_space = 11
    ace_state_space = 2
    action_space = 2

    new_q = np.zeros((player_state_space, dealer_state_space, ace_state_space, action_space))

    for state, value in q.items():

        player_score, dealer_card, ace = state
        new_q[player_score][dealer_card][int(ace)][0] = value[0]
        new_q[player_score][dealer_card][int(ace)][1] = value[1]

    return new_q

def plot_policy(policy):

    player_range = np.arange(11, 22)  # consider from 11 to 21
    dealer_range = np.arange(1, 11)  # consider from 1 to 10

    policy_na = change_policy_shape(policy, 0)
    policy_wa = change_policy_shape(policy, 1)

    # Get colors
    cmap = cm.get_cmap("Paired")
    colors = list([cmap(0.2), cmap(0.8)])
    label = ["Stand", "Hit"]

    # Plot results
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.pcolor(dealer_range, player_range,
               policy_wa, label=label,
               cmap=mpl.colors.ListedColormap(colors))
    plt.axis([dealer_range.min(), dealer_range.max(),
              player_range.min(), player_range.max()])
    col_bar = plt.colorbar()
    col_bar.set_ticks([0.25, 0.75])
    col_bar.set_ticklabels(label)
    plt.grid()
    plt.xlabel("Dealer Score")
    plt.ylabel("Player Score")
    plt.title("Optimal Policy With an Ace ($\pi_*$)")

    plt.subplot(122)
    plt.pcolor(dealer_range, player_range,
               policy_na, label=label,
               cmap=mpl.colors.ListedColormap(colors))
    plt.axis([dealer_range.min(), dealer_range.max(),
              player_range.min(), player_range.max()])
    col_bar = plt.colorbar()
    col_bar.set_ticks([0.25, 0.75])
    col_bar.set_ticklabels(label)
    plt.grid()
    plt.xlabel("Dealer Score")
    plt.ylabel("Player Score")
    plt.title("Optimal Policy Without an Ace ($\pi_*$)")
    plt.show()