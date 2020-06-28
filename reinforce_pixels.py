#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator


# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b



class Network:
    def __init__(self, env, args):

        # The inputs have shape `env.state_shape`,
        # and the model should produce probabilities of `env.actions` actions.
        # You can use for example one hidden layer with `args.hidden_layer` and non-linear activation.
        inputs = tf.keras.layers.Input(shape=env.state_shape)

        model = tf.keras.models.Sequential()

        # 1
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=env.state_shape))  # 32,32,3
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))  # 16

        # 2
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))  # 8

        # 3
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))  # 4
        model.add(tf.keras.layers.BatchNormalization())

        # 2x dense
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))

        # last dense
        model.add(tf.keras.layers.Dense(units=env.actions, activation=tf.nn.softmax))
        self.model = model

        # Use Adam optimizer with given `args.learning_rate`.
        o = tf.optimizers.Adam(lr=args.learning_rate)
        self.model.compile(optimizer=o, loss='mse')  # mse??

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        # Train the model using the states, actions and observed returns by calling `train_on_batch`.
        # States  [ batch , 4 ]
        # Actions [ batch , 2 ]
        # returns [ batch , 1 ]

        onehot_actions = np.zeros((actions.size, 2), dtype=np.int32)
        onehot_actions[np.arange(actions.size), actions] = 1
        # (246, 2)
        # (95, 1)
        assert onehot_actions.shape[1] == 2
        self.model.train_on_batch(x=states, y=onehot_actions, sample_weight=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        # Predict distribution over actions for the given input states
        # using the `predict_on_batch` method and calling `.numpy()` on the result to return a NumPy array.
        results = self.model.predict_on_batch(x=states)
        results = results.numpy()
        return results


def calculate_returns(rewards, gamma=0.99, subs_mean=False):
    # apply discount [1, 0.99, 0.98, 0.97]
    returns = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    # cumsum from back:
    # [1,1,1,1,1] -> [5,4,3,2,1]
    # [1, 0.99, 0.98, 0.97]   --> [5, 4, 3.1, 2.2, 1.3]
    returns = np.cumsum(returns[::-1])[::-1]
    if subs_mean:
        returns -= returns.mean()
    return returns


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################

    # epochs
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")

    # batch_size
    parser.add_argument("--batch_size", default=20, type=int, help="Number of episodes to train on.")

    # gamma
    parser.add_argument("--gamma", default=0.99, type=float, help="gamma for discount.")

    # lr
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")

    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################

    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create the environment
    env = cart_pole_pixels_evaluator.environment(seed=args.seed)
    possible_actions = list(range(env.actions))

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []

        # Batch over multiple episodes (failed / finished)
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:

                # render image
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # select action proportional to probability distribution given by the network
                action = np.random.choice(a=possible_actions, p=network.predict([state])[0])
                next_state, reward, done, _ = env.step(action)

                # append state action reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            # add results of this episode to the batches (every time different # ...)
            # what we do we create one giant array ... [1batch == action across 10 episodes]
            # return are 'zeroed' every time we fail
            batch_states += states
            batch_actions += actions
            batch_returns += calculate_returns(rewards, gamma=args.gamma, subs_mean=False).tolist()
        print(batch_actions)
        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
