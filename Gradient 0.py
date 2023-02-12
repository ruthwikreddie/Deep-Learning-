{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1d133a72",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "import torchvision as tv\n",
    "from torchvision import transforms, datasets\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1ff30450",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create random data between (-10, 10) and determine groundtruth\n",
    "sim_in = 20 * torch.rand((1000, 1)) - 10\n",
    "groundTruth = np.cos(sim_in)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c9eca80d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the number of parameters in a neural network\n",
    "def calcParams(inputModel):\n",
    "    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)\n",
    "    return val"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1f0f1633",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up NN for cos(x) training - 2 Hidden Layers, 32 parameters - Shallow Network\n",
    "class OptimizeNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.fc1 = nn.Linear(1, 7)\n",
    "        self.fc2 = nn.Linear(7, 6)\n",
    "        self.fc3 = nn.Linear(6, 1)\n",
    "\n",
    "    def forward(self, val):\n",
    "        val = F.relu(self.fc1(val))\n",
    "        val = F.relu(self.fc2(val))\n",
    "        val = self.fc3(val)\n",
    "        return val"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a7e0846b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up necessary auxilaries for neural net training\n",
    "optimNet = OptimizeNN()\n",
    "costFunc = nn.MSELoss()\n",
    "opt = optim.Adam(optimNet.parameters(), lr=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "56b8801e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def min_ratio(w):\n",
    "    count = 0\n",
    "    total = 0\n",
    "    for x in w:\n",
    "        if(x>0):\n",
    "            count += 1\n",
    "        total +=1\n",
    "    return count/ total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5f6a2ff7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_grad_norm(model):\n",
    "\n",
    "    grads = []\n",
    "    for p in model.modules():\n",
    "        if isinstance(p, nn.Linear):\n",
    "            param_norm = p.weight.grad.norm(2).item()\n",
    "            grads.append(param_norm)\n",
    "\n",
    "    grad_mean = np.mean(grads) # compute mean of gradient norms\n",
    "    return grad_mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "75e66cc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "minRatio = []\n",
    "Loss = []\n",
    "train_count = 100\n",
    "for i in range(train_count):\n",
    "    EPOCHS = 100\n",
    "    lss = 0\n",
    "    # Set up necessary auxilaries for neural net training\n",
    "    optimNet = OptimizeNN()\n",
    "    costFunc = nn.MSELoss()\n",
    "    opt = optim.Adam(optimNet.parameters(), lr=0.001)\n",
    "    for epochIndex in range(EPOCHS):\n",
    "        optimNet.zero_grad()\n",
    "        output = optimNet(sim_in)\n",
    "        cost = costFunc(output, groundTruth)\n",
    "        with torch.no_grad():\n",
    "            lss += (costFunc(output, groundTruth))\n",
    "        cost.backward()\n",
    "        opt.step()\n",
    "    #print(calculate_grad_norm(optimNet))\n",
    "    Loss.append(lss/len(sim_in))\n",
    "    num_param = calcParams(optimNet)\n",
    "    \n",
    "    # Allocate Hessian size\n",
    "    H = torch.zeros((num_param, num_param))\n",
    "\n",
    "    y_hat = optimNet(sim_in)\n",
    "    y = sim_in\n",
    "    loss  = ((y_hat - y)**2).mean()\n",
    "    # Calculate Jacobian w.r.t. model parameters\n",
    "    J = torch.autograd.grad(loss, list(optimNet.parameters()), create_graph=True)\n",
    "    J = torch.cat([e.flatten() for e in J]) # flatten\n",
    "\n",
    "    # Fill in Hessian\n",
    "    num_param = calcParams(optimNet)\n",
    "    for i in range(num_param):\n",
    "        result = torch.autograd.grad(J[i], list(optimNet.parameters()), retain_graph=True)\n",
    "        H[i] = torch.cat([r.flatten() for r in result]) # flatten\n",
    "    w, v = np.linalg.eig(H)\n",
    "    minRatio.append(min_ratio(w))\n",
    "    #print(min_ratio(w))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "148e82a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_param = calcParams(optimNet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "71c49b76",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "H = torch.zeros((num_param, num_param))\n",
    "\n",
    "y_hat = optimNet(sim_in)\n",
    "y = sim_in\n",
    "loss  = ((y_hat - y)**2).mean()\n",
    "\n",
    "J = torch.autograd.grad(loss, list(optimNet.parameters()), create_graph=True)\n",
    "J = torch.cat([e.flatten() for e in J]) # flatten\n",
    "\n",
    "\n",
    "for i in range(num_param):\n",
    "    result = torch.autograd.grad(J[i], list(optimNet.parameters()), retain_graph=True)\n",
    "    H[i] = torch.cat([r.flatten() for r in result]) # flatten"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "ec5ac242",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Loss')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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HfcGgdXV1UpKStLhw4e5671FOAehgfNgPc5BaAj382AYhk6ePKkLL7xQERGtj/KhBciDiIgI9e3b1+pqBE1sbGxYvtA7Es5BaOA8WI9zEBrC+Ty01fLTiEHQAADAdghAAADAdghANhYdHa2lS5cqOjra6qrYFucgNHAerMc5CA12Og8MggYAALZDCxAAALAdAhAAALAdAhAAALAdAhAAALAdAlAHl5eXp5SUFMXExCgjI0OFhYUtln3llVc0btw49erVS7GxscrMzNRbb70VxNp2TL6cg48++kgjR45Ujx49dN555+mSSy7RU089FcTadky+nINzffzxx+rUqZMGDx7cvhW0CV/Ow/vvvy+Hw9Hs64svvghijTseX98LNTU1ysnJUXJysqKjozVgwABt2LAhSLVtZwY6rC1bthidO3c21q9fb+zbt8+YN2+ecf755xvffvutx/Lz5s0zVq1aZezatcs4cOCAsWjRIqNz587G7t27g1zzjsPXc7B7925j06ZNxt69e41Dhw4Zf/7zn40uXboYzz77bJBr3nH4eg4anThxwrjooouMrKwsY9CgQcGpbAfm63l47733DEnGl19+aTidTtfX2bNng1zzjsPMe+HGG280hg8fbhQUFBiHDh0yPv30U+Pjjz8OYq3bDwGoA7viiiuMWbNmuW275JJLjIULF3q9j7S0NGP58uWBrpptBOIc3HTTTcb06dMDXTXbMHsOpkyZYixevNhYunQpASgAfD0PjQHohx9+CELt7MHXc/Dmm28acXFxRmVlZTCqF3R0gXVQtbW1Ki4uVlZWltv2rKws7dy506t91NfX6+TJk+revXt7VLHDC8Q52LNnj3bu3KkxY8a0RxU7PLPn4Pnnn9fBgwe1dOnS9q6iLfjzXhgyZIgSExN1zTXX6L333mvPanZoZs7Btm3bNGzYMD366KPq06ePfvGLX+iBBx7Qv/71r2BUud1xM9QO6tixY6qrq1N8fLzb9vj4eJWXl3u1jyeeeEI//vijbr311vaoYofnzzno27evjh49qrNnz2rZsmW655572rOqHZaZc/DVV19p4cKFKiwsVKdOfEQGgpnzkJiYqHXr1ikjI0M1NTX685//rGuuuUbvv/++rrrqqmBUu0Mxcw6++eYbffTRR4qJidGrr76qY8eOac6cOTp+/HiHGAfEu7uDczgcbj8bhtFsmyebN2/WsmXL9Nprr6l3797tVT1bMHMOCgsLderUKX3yySdauHChLr74Yk2dOrU9q9mheXsO6urqNG3aNC1fvly/+MUvglU92/DlvTBw4EANHDjQ9XNmZqYOHz6sxx9/nADkB1/OQX19vRwOh/7yl7+47rD+5JNP6pZbbtGaNWt03nnntXt92xMBqIPq2bOnIiMjmyX7ioqKZv8BNLV161bdfffd+tvf/qZrr722PavZoflzDlJSUiRJl112mf7nf/5Hy5YtIwCZ4Os5OHnypP75z39qz549uu+++yQ1XAQMw1CnTp20Y8cOXX311UGpe0fiz3vhXFdeeaU2btwY6OrZgplzkJiYqD59+rjCjySlpqbKMAx99913+vnPf96udW5vjAHqoKKiopSRkaGCggK37QUFBRoxYkSLj9u8ebPuvPNObdq0STfccEN7V7NDM3sOmjIMQzU1NYGuni34eg5iY2P1+eefq6SkxPU1a9YsDRw4UCUlJRo+fHiwqt6hBOq9sGfPHiUmJga6erZg5hyMHDlS33//vU6dOuXaduDAAUVERKhv377tWt+gsG78Ndpb45TH5557zti3b5+RnZ1tnH/++UZpaalhGIaxcOFCY8aMGa7ymzZtMjp16mSsWbPGbdrpiRMnrDqEsOfrOXj66aeNbdu2GQcOHDAOHDhgbNiwwYiNjTVycnKsOoSw5+s5aIpZYIHh63l46qmnjFdffdU4cOCAsXfvXmPhwoWGJOPll1+26hDCnq/n4OTJk0bfvn2NW265xfjv//5v44MPPjB+/vOfG/fcc49VhxBQBKAObs2aNUZycrIRFRVlDB061Pjggw9cv7vjjjuMMWPGuH4eM2aMIanZ1x133BH8incgvpyDP/zhD8all15qdOnSxYiNjTWGDBli5OXlGXV1dRbUvOPw5Rw0RQAKHF/Ow6pVq4wBAwYYMTExRrdu3YxRo0YZb7zxhgW17lh8fS/s37/fuPbaa43zzjvP6Nu3r7FgwQLj9OnTQa51+3AYhmFY2QIFAAAQbIwBAgAAtkMAAgAAtkMAAgAAtkMAAgAAtkMAAgAAtkMAAgAAtkMAAgAAtkMAAgAAtkMAAhA0Y8eOVXZ2ttflS0tL5XA4VFJS0m51CubztOT999+Xw+HQiRMnLHl+wI4IQABMu/POO+VwODRr1qxmv5szZ44cDofuvPNO17ZXXnlFDz/8sNf7T0pKktPpVHp6eiCq65exY8fK4XDI4XAoKipKAwYM0KJFi3y+Ua2nEDhixAg5nU63u24DaF8EIAB+SUpK0pYtW/Svf/3Lte3MmTPavHmz+vXr51a2e/fu6tq1q9f7joyMVEJCgjp16hSw+vpj5syZcjqd+vrrr/Xoo49qzZo1WrZsmd/7jYqKUkJCghwOh/+VBOAVAhAAvwwdOlT9+vXTK6+84tr2yiuvKCkpSUOGDHEr27T1o3///nrkkUd01113qWvXrurXr5/WrVvn+n3TrqnGrqK33npLQ4YM0Xnnnaerr75aFRUVevPNN5WamqrY2FhNnTpVp0+fdu1n+/btGjVqlC644AL16NFD//Zv/6aDBw/6fKxdunRRQkKC+vXrp5tvvlnjxo3Tjh07XL+vrKzU1KlT1bdvX3Xp0kWXXXaZNm/e7Pr9nXfeqQ8++EC///3vXa1JpaWlHrvAXn75ZV166aWKjo5W//799cQTT/hcXwAtIwAB8Nu///u/6/nnn3f9vGHDBt11111ePfaJJ57QsGHDtGfPHs2ZM0ezZ8/WF1980epjli1bpqefflo7d+7U4cOHdeutt2r16tXatGmT3njjDRUUFOg///M/XeV//PFHLViwQP/1X/+ld955RxEREbrppptUX19v7oAlffbZZ/r444/VuXNn17YzZ84oIyND//jHP7R3717de++9mjFjhj799FNJ0u9//3tlZma6WpKcTqeSkpKa7bu4uFi33nqrbrvtNn3++edatmyZlixZohdeeMF0fQE0YfXt6AGErzvuuMOYNGmScfToUSM6Oto4dOiQUVpaasTExBhHjx41Jk2aZNxxxx2u8mPGjDHmzZvn+jk5OdmYPn266+f6+nqjd+/extq1aw3DMIxDhw4Zkow9e/YYhmEY7733niHJePvtt12Pyc3NNSQZBw8edG37zW9+Y1x33XUt1ruiosKQZHz++ecen8eTMWPGGJ07dzbOP/98IyoqypBkREREGC+99FKrf6Prr7/euP/++1v8G5x7XD/88INhGIYxbdo0Y9y4cW5l/u///b9GWlpaq88FwHu0AAHwW8+ePXXDDTfoT3/6k55//nndcMMN6tmzp1ePvfzyy13fOxwOJSQkqKKiwuvHxMfHq0uXLrrooovctp27j4MHD2ratGm66KKLFBsbq5SUFElSWVmZV3Vs9Otf/1olJSUqKirSrbfeqrvuuks333yz6/d1dXVauXKlLr/8cvXo0UM/+9nPtGPHDp+fZ//+/Ro5cqTbtpEjR+qrr75SXV2dT/sC4FlojCwEEPbuuusu3XfffZKkNWvWeP24c7uQpIYQ1FbX1LmPcTgcbe5j4sSJSkpK0vr163XhhReqvr5e6enpqq2t9bqekhQXF6eLL75YkrRx40Zdeumleu6553T33XdLaujOe+qpp7R69WpddtllOv/885Wdne3z8xiG0WxAtGEYPu0DQOtoAQIQEOPHj1dtba1qa2t13XXXWV0dl8rKSu3fv1+LFy/WNddco9TUVP3www9+77dz58767W9/q8WLF7sGXBcWFmrSpEmaPn26Bg0apIsuukhfffWV2+OioqLabMVJS0vTRx995LZt586d+sUvfqHIyEi/6w6AAAQgQCIjI7V//37t378/pC7S3bp1U48ePbRu3Tp9/fXXevfdd7VgwYKA7HvatGlyOBzKy8uTJF188cUqKCjQzp07tX//fv3mN79ReXm522P69++vTz/9VKWlpTp27JjH1q77779f77zzjh5++GEdOHBAf/rTn/T000/rgQceCEi9ARCAAARQbGysYmNjra6Gm4iICG3ZskXFxcVKT0/X/Pnz9dhjjwVk31FRUbrvvvv06KOP6tSpU1qyZImGDh2q6667TmPHjlVCQoImT57s9pgHHnhAkZGRSktLU69evTyODxo6dKj++te/asuWLUpPT9dDDz2kFStWuC0qCcA/DoOOZQAAYDO0AAEAANshAAEAANshAAEAANshAAEAANshAAEAANshAAEAANshAAEAANshAAEAANshAAEAANshAAEAANshAAEAANv5/wGsq5Rkhdmf+QAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(minRatio, Loss)\n",
    "plt.xlabel('Minimal Ratio')\n",
    "plt.ylabel('Loss')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95b33935",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
