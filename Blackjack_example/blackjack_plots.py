import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import copy
from matplotlib import cm
import numpy as np
from typing import Callable, Any, Tuple

# Types
PlayerTotal = int
DealerCard = int
UsableAce = bool

State = Tuple[PlayerTotal, DealerCard, UsableAce]

# Dims
player_state_space_dim = 32
action_space_dim = 2
dealer_state_space_dim = 11
ace_state_space_dim = 2

# changing policy repr
def change_policy_to_np_array(policy, policy_type: str = "Function") -> np.ndarray:
    '''change policy to numpy array depending on policy_type'''

    if policy_type == 'dict':
        return change_dict_policy_to_np_array(policy)

    # assume its a function
    else:
        return change_fct_policy_to_np(policy)

def change_fct_policy_to_np(policy: Callable) -> np.ndarray:
    '''function to change function repr of policy to np arr'''

    policy_arr = np.ones(
        (player_state_space_dim,
        dealer_state_space_dim,
        ace_state_space_dim,
        action_space_dim,
        ))

    for i in range(player_state_space_dim):
        for j in range(dealer_state_space_dim):
            for k in range(ace_state_space_dim):

                policy_arr[i,j,k] = policy((i, j, k))

    return policy_arr

def change_dict_policy_to_np_array(policy: dict) -> np.ndarray:
    '''function to change dict repr of policy to np array'''

    policy_arr = np.ones(
        (player_state_space_dim,
        dealer_state_space_dim,
        ace_state_space_dim,
        action_space_dim
        ))

    # default to always hit if not seen in dict
    policy_arr[:, :, :] = [0, 1]

    for state, action_prob in policy.items():
        player_score, dealer_card, ace = state

        policy_arr[player_score, dealer_card, int(ace)] = action_prob


    return policy_arr

def argmax_policy(policy:np.ndarray, policy_type = "Function") -> np.ndarray:
    '''
    function to turn policy completely greedy by looking for
    argmax action over action dimension
    '''

    return np.argmax(policy, axis = 3)


# changing Value function representations
def change_dict_value_fct_to_np(v: dict) -> np.ndarray:
    '''
    Changes state values from dictionaries to np arrays
    '''

    new_v = np.zeros(
        (player_state_space_dim,
        dealer_state_space_dim,
        ace_state_space_dim
        ))

    for state, value in v.items():

        player_score, dealer_card, ace = state
        new_v[player_score, dealer_card, int(ace)] = value

    return new_v

def change_dict_q_value_to_np(q: dict) -> np.ndarray:

    '''
    Change action-values from dictionaries to np arrays
    '''

    new_q = np.zeros(
        (player_state_space_dim,
        dealer_state_space_dim,
        ace_state_space_dim,
        action_space_dim
        ))

    for state, value in q.items():

        player_score, dealer_card, ace = state
        new_q[player_score, dealer_card, int(ace), 0] = value[0]
        new_q[player_score, dealer_card, int(ace), 1] = value[1]

    return new_q

def plot_values(state_val):

    state_val = change_dict_value_fct_to_np(state_val)

    # without usable ace
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize=(12, 8))

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
    ax.set_title("State Value ($V_\pi$) Without Ace")
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("State Value")
    plt.show()

    # With usable ace
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize=(12, 8))

    X, Y = np.meshgrid(dealer_range, player_range)

    Z = state_val[11:22, 1:11, 1].reshape(X.shape)  # usable ace

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=1,
                    rstride=1, cstride=1)
    ax.set_title("State Value ($V_\pi$) With Ace")
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("State Value")
    plt.show()

def plot_policy(policy: Any):

    if type(policy) != np.ndarray:
        policy = change_fct_policy_to_np(policy=policy)

    fig, axs, = plt.subplots(1, 2, figsize = (15, 6))

    extent = -1, 9, -1, 11  # the x tick limits
    x_range = np.arange(10)

    # first plot
    surf = axs[0].imshow(
        np.flip(policy[10:22,1:11,0, 1], axis=0),  # flip so that low player card is at the bottom not top
        cmap=cm.Blues,
        vmin=0,
        vmax=1,
        interpolation='nearest',
        extent=extent
    )

    plt.sca(axs[0])  # set current axis
    plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.yticks(np.arange(12), (str(i) for i in range(10, 22, 1)))
    axs[0].set_xlabel('Dealer Card Showing')
    axs[0].set_ylabel('Player Total')
    axs[0].set_title("Policy ($\pi$) Without Ace")
    axs[0].grid()

    col_bar = fig.colorbar(surf, ax=axs[0])
    col_bar.set_ticks([0,0.5, 1])
    col_bar.set_ticklabels(['Stay (0)', 'Random (0.5)', 'Hit (1)'])
    col_bar.ax.invert_yaxis()  # flip the y axis

    # second plot
    surf = axs[1].imshow(
        np.flip(policy[10:22,1:11,1, 1], axis = 0),  # flip so that low player card is at the bottom not top
        cmap=cm.Blues,
        vmin=0,
        vmax=1,
        interpolation='nearest',
        extent=extent
    )

    plt.sca(axs[1])  # set current axis
    plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.yticks(np.arange(12), (str(i) for i in range(10, 22, 1)))
    axs[1].set_xlabel('Dealer Card Showing')
    axs[1].set_ylabel('Player Total')
    axs[1].set_title("Policy ($\pi$) With Ace")
    axs[1].grid()

    col_bar = fig.colorbar(surf, ax=axs[1])
    col_bar.set_ticks([0,0.5, 1])
    col_bar.set_ticklabels(['Stay (0)', 'Random (0.5)', 'Hit (1)'])
    col_bar.ax.invert_yaxis()  # flip the y axis

    plt.show()
