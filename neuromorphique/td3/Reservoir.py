import torch
import torch.nn as nn

def to_sparse(tensor, density):
    """ correspond aux interconnections entre neurones dans le réservoir
    on va vouloir supprimer des interactions entre neurones
    en en fixant un certain nombre (choisis aleatoirement) à zero"""
    return tensor * (torch.rand_like(tensor) <= density).type(tensor.dtype)

def random_matrix(size):
    return torch.rand(size)

class reservoir(torch.nn.Module):
    """
    Implements a reservoir.

    Parameters:
      - input_size: size of the input
      - reservoir_size: number of units in the reservoir
      - contractivity_coeff: spectral radius for the reservoir matrix
      - density: density of the reservoir matrix, from 0 to 1.
      - scale_in: scaling of the input-to-reservoir matrix
      - f: activation function for the state transition function
    """

    def __init__(self, input_size, reservoir_size, contractivity_coeff=0.9, density=1.0, scale_in=1.0, f=torch.relu):
        super(reservoir, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.contractivity_coeff = contractivity_coeff # détermine la mémoire du réservoir -> la plus grande (en norme) vp de la matrice du réservoir
        # (détermine à quel point on se rappelle de l'input précédent)
        self.density = density # décide à quel point les neurones sont connectés les une aux autres (i.e. si la matrice est sparse)
        self.scale_in = scale_in
        self.f = f # relu ou th

        # poids d'entrée
        self.W_in = random_matrix((reservoir_size, input_size)) * 2 - 1
        #( valeurs entre -1 et 1)
        #matrice du réservoir
        self.W_hat = random_matrix((reservoir_size, reservoir_size)) * 2 - 1
        self.W_hat = to_sparse(self.W_hat, density)

        self.W_in = scale_in * self.W_in # pour borner (What?)

        # Prescale W_hat
        self.W_hat = self._rescale_contractivity(self.W_hat) # pour borner (What?)

        # Register as parameters
        self.W_in = nn.Parameter(self.W_in, requires_grad=False)
        self.W_hat = nn.Parameter(self.W_hat, requires_grad=False)

    def forward(self, input, initial_state=None):
        """
        Compute the reservoir states for the given sequence.

        Parameters:
          - input: Input sequence of shape (seq_len, input_size)

        Returns: a tensor of shape (seq_len, reservoir_size)
        """
        x = torch.zeros((input.size(0), self.reservoir_size), device=self.W_hat.device) # pourraient etre dans un etat initial différent

        # @ = torch.mm = mult matricielle
        # on utilise ici self.f -> la fonction d'activation
        if initial_state is not None:
            x[0,:] = self.f( self.W_in @ input[0,:] + self.W_hat @ initial_state )
        else:
            x[0,:] = self.f( self.W_in @ input[0,:] )

        for i in range(1, len(input)):
            # memoire de l'instant precedant :self.W_hat @ x[i-1]
            # ceci explique rayon_spectral(W_hat) <= 1, sinon memoire non bornee
            x[i,:] = self.f( self.W_in @ input[i,:] + self.W_hat @ x[i-1] )
        return x

    def _rescale_contractivity(self, W):
        coeff = self.contractivity_coeff
        return W * coeff / (W.eig()[0].abs().max())

