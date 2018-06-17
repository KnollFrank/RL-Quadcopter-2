from keras import layers, initializers, models, optimizers
from keras import backend as K
from keras import regularizers

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, learning_rate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.learning_rate = learning_rate

        self.build_model()

    def create_block(self, units, inputs):
        net = layers.Dense(units = units, kernel_regularizer = regularizers.l2(0.01))(inputs)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        return net

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = self.create_block(units = 32, inputs = states)
        net_states = self.create_block(units = 64, inputs = net_states)

        # Add hidden layer(s) for action pathway
        net_actions = self.create_block(units = 32, inputs = actions)
        net_actions = self.create_block(units = 64, inputs = net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = self.create_block(units = 32, inputs = net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=1e-4), name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
