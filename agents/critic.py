from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K
from keras import regularizers

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        # FK-TODO: units=300
        # net_states = layers.Dense(units=32, activation='relu')(states)
        # FK-TODO: units=400
        # net_states = layers.Dense(units=64, activation='relu')(net_states)
        kernel_l2_reg = 1e-5
        net_states = layers.Dense(units=300, kernel_regularizer=regularizers.l2(kernel_l2_reg))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)

        net_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(kernel_l2_reg))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)

        # Add hidden layer(s) for action pathway
        # net_actions = layers.Dense(units=32, activation='relu')(actions)
        # net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions = layers.Dense(units=400, kernel_regularizer=regularizers.l2(kernel_l2_reg))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(1e-2)(net_actions)

        # FK-TODO: Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # FK-TODO: Add more layers to the combined network if needed
        net = layers.Dense(units=200, kernel_regularizer=regularizers.l2(kernel_l2_reg))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)