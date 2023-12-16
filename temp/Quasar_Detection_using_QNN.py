{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ap7nRE9m6rXh"
      },
      "source": [
        "# Detection of Quasar using Quantum Machine Learning\n",
        "\n",
        "\n",
        "Then the data is encoded and trained using a quantum circuit using the parameterized quantum circuit taken from the paper [Expressibilitty and entangling capability of parameterized quantum circuit for hybrid quantum-classical algorithms](https://arxiv.org/abs/1905.10876).\n",
        "\n",
        "-----\n",
        "\n",
        "With the high expressibility of the circuit, the model is trained and tested with Cross-Entropy as the Loss function and **L-BFGS** algorithm for optimization.\n",
        "\n",
        "Dataset is taken from The Sloan Digital Sky Survey Quasar Catalog: sixteenth data release (DR16Q)\n",
        "\n",
        "The accuracy of the model is $94\\%$ via a quantum machine learning model \n",
        "\n",
        "\n",
        "This algorithm is realized in **PyTorch** and **Qiskit** machine learning module."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4_4Ylck69OK9"
      },
      "source": [
        "## Pre-Processing of Image\n",
        "\n",
        "\n",
        "Import necessary Libraries for Image pre-processing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mgpX3QNh9Rwj"
      },
      "outputs": [],
      "source": [
        "from PIL import Image\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fCnlSHdnCWkO"
      },
      "source": [
        "-----\n",
        "-----\n",
        "\n",
        "Import Basic python packages and the two python helper scirpts that we made to make our life easier.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "#Image processing\n",
        "from PIL import Image\n",
        "\n",
        "#Calling Image from the path, and file name. \n",
        "#i is the number labeling of the data\n",
        "#some of the files does not have name\n",
        "def callImage(i,path,name):\n",
        "    x1 = Image.open(\n",
        "        path+str(name)+str(i)+'.jpg').convert('L');\n",
        "    y1 = np.asarray(x1.getdata(), dtype=np.float64).reshape((x1.size[1], x1.size[0]));\n",
        "    y_dat1 = np.asarray(y1, dtype=np.uint8)     \n",
        "    return y_dat1\n",
        "\n",
        "#Resize image into n x n pixel\n",
        "def imageResize(data,pixel):\n",
        "    image = Image.fromarray(data,'L')\n",
        "    image= image.resize((pixel, pixel))\n",
        "    image=np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))\n",
        "    image=np.asarray(image, dtype=np.uint8)    \n",
        "    return image\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "#Making MxN partition\n",
        "def imagePartition(data,M,N):\n",
        "    tiles = [data[x:x+M,y:y+N] for x in range(0,data.shape[0],M) for y in range(0,data.shape[1],N)]\n",
        "    return tiles\n",
        "\n",
        "def imageBinarize(data):\n",
        "    # specify a threshold 0-255\n",
        "    threshold = 75\n",
        "    # make all pixels < threshold black\n",
        "    bidata = 1.0 * (data > threshold)\n",
        "    return bidata"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import qiskit\n",
        "from qiskit import transpile, assemble\n",
        "from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute\n",
        "from qiskit import BasicAer, Aer, execute\n",
        "from qiskit.quantum_info import state_fidelity\n",
        "from qiskit.visualization import *\n",
        "from qiskit.quantum_info.operators import Operator\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from qiskit.circuit.parameter import Parameter\n",
        "import torch\n",
        "from torch.autograd import Function\n",
        "from torchvision import datasets, transforms\n",
        "import torch.optim as optim\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "import qiskit\n",
        "from qiskit import transpile, assemble\n",
        "from qiskit.visualization import *\n",
        "\n",
        "\n",
        "nqubits=6\n",
        "\n",
        "def normlaizeData(data):\n",
        "    #Create Array of pixel value\n",
        "    testdata=data\n",
        "    arr_data=testdata.flatten()/max(testdata.flatten());\n",
        "    encoding_data= np.array([np.round(x,6) for x in arr_data]);\n",
        "    sum_const=np.sqrt(sum(encoding_data*encoding_data))\n",
        "    encoding_norm=encoding_data/sum_const\n",
        "    return encoding_norm\n",
        "\n",
        "# Choose on PQC from Hannah  Sim https://arxiv.org/pdf/1905.10876.pdf circuit 15\n",
        "\n",
        "def circuit15(qc,theta):\n",
        "    #circuit 15\n",
        "    #theta is list of the parameters\n",
        "    #theta length is (8)L\n",
        "    #L is the number of repeatation\n",
        "    nqubits=6\n",
        "    qr = QuantumRegister(nqubits)\n",
        "    qc = QuantumCircuit(qr, name='PQC')\n",
        "\n",
        "    count=0\n",
        "\n",
        "\n",
        "    for i in range(nqubits):\n",
        "        qc.ry(theta[count],i)\n",
        "        count=count+1\n",
        "    for i in range(nqubits-1):\n",
        "        qc.cx(i,i+1)\n",
        "    \n",
        "    qc.cx(0,nqubits-1)\n",
        "    for i in range(nqubits):\n",
        "        qc.ry(theta[count],i)\n",
        "        count=count+1    \n",
        "    for i in range(nqubits-1):\n",
        "        qc.cx(i+1,i)\n",
        "    qc.cx(nqubits-1,0)\n",
        "    qc.to_instruction()\n",
        "    return qc\n",
        "# Choose on PQC from Hannah  Sim https://arxiv.org/pdf/1905.10876.pdf circuit 15\n",
        "\n",
        "def encoding(qc,theta,L):\n",
        "    #circuit 15\n",
        "    #theta is list of the parameters\n",
        "    #theta length is (8)L\n",
        "    #L is the number of repeatation\n",
        "    nqubits=6\n",
        "    qr = QuantumRegister(nqubits)\n",
        "    qc = QuantumCircuit(qr, name='Embed')\n",
        "\n",
        "    count=0\n",
        "    for i in range(nqubits):\n",
        "        qc.h(i)\n",
        "        \n",
        "    for l in range(L):\n",
        "        for i in range(nqubits):\n",
        "            qc.ry(theta[count],i)\n",
        "            count=count+1\n",
        "        for i in range(nqubits-1):\n",
        "            qc.cx(i,i+1)\n",
        "        \n",
        "        qc.cx(nqubits-1,0)\n",
        "        for i in range(nqubits):\n",
        "            qc.ry(theta[count],i)\n",
        "            count=count+1    \n",
        "        for i in range(nqubits-1):\n",
        "            qc.cx(i+1,i)\n",
        "        qc.cx(0,nqubits-1)\n",
        "        \n",
        "    qc.to_instruction()\n",
        "    return qc\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "# mapping the data\n",
        "# mapping is taken from https://arxiv.org/pdf/2003.09887.pdf\n",
        "def binary(x):\n",
        "    return ('0'*(6-len('{:b}'.format(x, '#010b') ))+'{:b}'.format(x, '#010b'))\n",
        "def firsttwo(x):\n",
        "    return x[:2]\n",
        "parity = lambda x: firsttwo(binary(x)).count('1') % 2   "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "YMKRo4jl5cI7"
      },
      "outputs": [],
      "source": [
        "target_o = [1 for i in range(25)]+[0 for i in range(25)]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "z7EuQ86NC1Bp"
      },
      "source": [
        "## Loading the dataset\n",
        "\n",
        "Load all the data files from the respective data folder and use the helper function to divide the images into smaller parts to make it easier for the QNN to read and make sense of."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "j0hhVp4p5cI_"
      },
      "outputs": [],
      "source": [
        "# from ImageRead import callImage, imageResize, imageBinarize  \n",
        "\n",
        "\n",
        "pathY=r'dataset/qso/'\n",
        "pathN=r'dataset/no_qso/'\n",
        "nameN=''\n",
        "nameY=''\n",
        "\n",
        "inputY=[imageResize(callImage(i+1,pathY,nameY),16) for i in range(25)]\n",
        "inputN=[imageResize(callImage(i+1,pathN,nameN),16) for i in range(25)]\n",
        "input_combine = inputY+inputN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u17hzIPlDNEr"
      },
      "source": [
        "Randomize and Re-Shuffle the data set to make it better to train."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PWO4DFzv5cJK"
      },
      "outputs": [],
      "source": [
        "np.random.seed(0)\n",
        "idx=np.array([int(i) for i in range(50)]).flatten()\n",
        "\n",
        "\n",
        "np.random.shuffle(idx)\n",
        "\n",
        "dataInput = list(input_combine[i] for i in idx )\n",
        "dataTarget = list( imageBinarize(input_combine[i]) for i in idx )\n",
        "\n",
        "data_target_o=list( target_o[i] for i in idx )"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Cj4u8emLDUKL"
      },
      "source": [
        "## Visualize\n",
        "\n",
        "Display the images from the folder after all the pre-processing and ready to be feeded to the QNN."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cjKji2Xa5cJM",
        "outputId": "ddc756ec-e181-48e6-fd24-b6c0e39d77aa"
      },
      "outputs": [],
      "source": [
        "n_samples_show = 10\n",
        "fig, axes = plt.subplots(nrows=2, ncols=n_samples_show, figsize=(20, 6))\n",
        "\n",
        "for i in range(n_samples_show):\n",
        "\n",
        "    axes[0,i].imshow(dataInput[i], cmap='gray')\n",
        "    axes[0,i].set_xticks([])\n",
        "    axes[0,i].set_yticks([])\n",
        "    axes[1,i].imshow(dataInput[i+5], cmap='gray')\n",
        "    axes[1,i].set_xticks([])\n",
        "    axes[1,i].set_yticks([])    \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aBcnyyP-DeTG"
      },
      "source": [
        "## The Quantum Neural Network\n",
        "\n",
        "\n",
        "Import necessary files for the training."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "trdX82lA5cJX"
      },
      "outputs": [],
      "source": [
        "from qiskit import Aer, QuantumCircuit\n",
        "from qiskit.circuit.parameter import Parameter\n",
        "from qiskit_machine_learning.connectors import TorchConnector\n",
        "from qiskit.utils import QuantumInstance\n",
        "from torch.nn import Linear, CrossEntropyLoss, MSELoss\n",
        "from torch.optim import LBFGS, SGD,Adam\n",
        "from PyFiles.QNN import circuit15, encoding, parity\n",
        "from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap\n",
        "from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC\n",
        "from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B\n",
        "from qiskit_machine_learning.neural_networks import SamplerQNN\n",
        "from qiskit.primitives import Sampler\n",
        "\n",
        "\n",
        "sampler = Sampler()\n",
        "\n",
        "\n",
        "\n",
        "# Model for LBFGS\n",
        "# Combining the circuit together with CircuitQNN\n",
        "np.random.seed(3)\n",
        "\n",
        "\n",
        "nqubits=6\n",
        "num_inputs=256\n",
        "qc = QuantumCircuit(nqubits)\n",
        "\n",
        "# Encoding\n",
        "param_x=[];\n",
        "for i in range(num_inputs):\n",
        "    param_x.append(Parameter('x'+str(i)))\n",
        "for i in range(8):\n",
        "    param_x.append(np.pi/2)\n",
        "\n",
        "\n",
        "feature_map = encoding(qc,param_x,22)\n",
        "\n",
        "\n",
        "# Optimzing circuit PQC\n",
        "param_y=[];\n",
        "for i in range(nqubits*2):\n",
        "    param_y.append(Parameter('θ'+str(i)))\n",
        "\n",
        "ansatz=circuit15(qc,param_y)\n",
        "\n",
        "qc.append(feature_map, range(nqubits))\n",
        "qc.append(ansatz, range(nqubits))\n",
        "\n",
        "# qnn2 = CircuitQNN(qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, \n",
        "#                   interpret=parity, output_shape=2, quantum_instance=qi)\n",
        "qnn2 = SamplerQNN(circuit= qc,  input_params=feature_map.parameters, weight_params=ansatz.parameters, \n",
        "                  interpret=parity, output_shape=2, sampler= sampler)\n",
        "initial_weights = 0.1*(2*np.random.rand(qnn2.num_weights) - 1)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YxOGd0gn5cJ_"
      },
      "source": [
        "# Learning Rate 0.05\n",
        "\n",
        "Training the model with a Learning Rate of 0.05"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FcnqOwQv5cKK"
      },
      "outputs": [],
      "source": [
        "# define optimizer and loss function\n",
        "\n",
        "from PyFiles.QNN import normlaizeData\n",
        "\n",
        "\n",
        "model2 = TorchConnector(qnn2, initial_weights)\n",
        "\n",
        "optimizer = LBFGS(model2.parameters(),lr=0.05)\n",
        "f_loss = CrossEntropyLoss()\n",
        "\n",
        "X= [normlaizeData(dataInput[i].flatten()) for i in range(50)]\n",
        "y01= [data_target_o[i] for i in range(50)]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "K0iuvuUi5cKM",
        "outputId": "47817de0-799c-4d9a-b9ed-2bf90f3d4060"
      },
      "outputs": [],
      "source": [
        "from torch import Tensor\n",
        "# traning model accuracy\n",
        "y_predict = []\n",
        "for x in X:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy:', sum(y_predict == np.array(y01))/len(np.array(y01)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Fm1Jj2XKENP_"
      },
      "source": [
        "This accuracy without applying any optimizer and just running on the first instance, as expected that the model is correct half of the time, even if you randomly classify the images you'll get the accuracy near $50\\%$ , so that's what we see here."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wC9qiJET5cKO"
      },
      "outputs": [],
      "source": [
        "# define optimizer and loss function\n",
        "\n",
        "model2 = TorchConnector(qnn2, initial_weights)\n",
        "\n",
        "optimizer = LBFGS(model2.parameters(),lr=0.05)\n",
        "f_loss = CrossEntropyLoss()\n",
        "\n",
        "X= [normlaizeData(dataInput[i].flatten()) for i in range(50)]\n",
        "y01= [data_target_o[i] for i in range(50)]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LEQQKM__5cKQ",
        "outputId": "49afb082-5339-49e6-a86a-11a16d7c1282"
      },
      "outputs": [],
      "source": [
        "from torch import Tensor\n",
        "# start training\n",
        "\n",
        "model2.train()    # set model to training mode\n",
        "\n",
        "# define objective function\n",
        "def closure():\n",
        "    optimizer.zero_grad()                                  # initialize gradient\n",
        "    loss = 0.0                                             # initialize loss    \n",
        "    for x, y_target in zip(X, y01):                        # evaluate batch loss\n",
        "        output = model2(Tensor(x)).reshape(1, 2)           # forward pass\n",
        "        loss += f_loss(output, Tensor([y_target]).long())\n",
        "    loss.backward()                                        # backward pass\n",
        "    print(loss.item())                                     # print loss\n",
        "    return loss\n",
        "\n",
        "# run optimizer\n",
        "optimizer.step(closure) \n",
        "optimizer.step(closure)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VWezyfvi5cKR",
        "outputId": "4aab0de1-7161-47eb-e6f4-69ecfef059e1"
      },
      "outputs": [],
      "source": [
        "# traning model accuracy\n",
        "y_predict = []\n",
        "for x in X:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy:', sum(y_predict == np.array(y01))/len(np.array(y01)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JVM_b1C7EBZm"
      },
      "source": [
        "Here you can see that the accuracy goes to $92\\%$ , which says that our model is good at classifying."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "x9ZsRoQh5cKV"
      },
      "source": [
        "# Learning rate 0.06\n",
        "\n",
        "Increasing the Learning Rate with $+0.01$ to $0.06$, we'll see how it affects the accuracy."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pSBdJB1h5cKX",
        "outputId": "52bd2aa4-df87-4f4e-f376-ae513d0e5fe3"
      },
      "outputs": [],
      "source": [
        "# define optimizer and loss function\n",
        "from torch import Tensor\n",
        "model2 = TorchConnector(qnn2, initial_weights)\n",
        "\n",
        "optimizer = LBFGS(model2.parameters(),lr=0.06)\n",
        "f_loss = CrossEntropyLoss()\n",
        "\n",
        "X= [normlaizeData(dataInput[i].flatten()) for i in range(50)]\n",
        "y01= [data_target_o[i] for i in range(50)]\n",
        "\n",
        "y_predict = []\n",
        "for x in X:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy:', sum(y_predict == np.array(y01))/len(np.array(y01)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8CbTEQys5cKZ",
        "outputId": "af452188-8426-4ba7-e912-35cb101c0aaf"
      },
      "outputs": [],
      "source": [
        "from torch import Tensor\n",
        "# start training\n",
        "\n",
        "model2.train()    # set model to training mode\n",
        "\n",
        "# define objective function\n",
        "def closure():\n",
        "    optimizer.zero_grad()                                  # initialize gradient\n",
        "    loss = 0.0                                             # initialize loss    \n",
        "    for x, y_target in zip(X, y01):                        # evaluate batch loss\n",
        "        output = model2(Tensor(x)).reshape(1, 2)           # forward pass\n",
        "        loss += f_loss(output, Tensor([y_target]).long())\n",
        "    loss.backward()                                        # backward pass\n",
        "    print(loss.item())                                     # print loss\n",
        "    return loss\n",
        "\n",
        "# run optimizer\n",
        "optimizer.step(closure)\n",
        "optimizer.step(closure)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "I33pz30p5cKb",
        "outputId": "4c015516-4b15-4a64-ae4a-c0312e6c47ee"
      },
      "outputs": [],
      "source": [
        "# traning model accuracy\n",
        "y_predict = []\n",
        "for x in X:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy:', sum(y_predict == np.array(y01))/len(np.array(y01)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qQUb7q5BEyqZ"
      },
      "source": [
        "We see the accuracy drops to $92\\%$ from $94\\%$."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1co0XUZm5cKd"
      },
      "source": [
        "# Learning Rate 0.07\n",
        "\n",
        "Increasing the accuracy even further by $0.01$"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HkMzQ3VM5cKe",
        "outputId": "a96725e8-4003-4381-fd06-93abbbd5af38"
      },
      "outputs": [],
      "source": [
        "# define optimizer and loss function\n",
        "from torch import Tensor\n",
        "model2 = TorchConnector(qnn2, initial_weights)\n",
        "\n",
        "optimizer = LBFGS(model2.parameters(),lr=0.07)\n",
        "f_loss = CrossEntropyLoss()\n",
        "\n",
        "X= [normlaizeData(dataInput[i].flatten()) for i in range(50)]\n",
        "y01= [data_target_o[i] for i in range(50)]\n",
        "\n",
        "y_predict = []\n",
        "for x in X:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy:', sum(y_predict == np.array(y01))/len(np.array(y01)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "43MQltGY5cKg",
        "outputId": "2e02fbdc-8f77-4281-8a9d-f15f1ebdc4bc"
      },
      "outputs": [],
      "source": [
        "from torch import Tensor\n",
        "# start training\n",
        "\n",
        "model2.train()    # set model to training mode\n",
        "\n",
        "# define objective function\n",
        "def closure():\n",
        "    optimizer.zero_grad()                                  # initialize gradient\n",
        "    loss = 0.0                                             # initialize loss    \n",
        "    for x, y_target in zip(X, y01):                        # evaluate batch loss\n",
        "        output = model2(Tensor(x)).reshape(1, 2)           # forward pass\n",
        "        loss += f_loss(output, Tensor([y_target]).long())\n",
        "    loss.backward()                                        # backward pass\n",
        "    print(loss.item())                                     # print loss\n",
        "    return loss\n",
        "\n",
        "# run optimizer\n",
        "optimizer.step(closure)\n",
        "optimizer.step(closure)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ge1Gn9LZ5cKi",
        "outputId": "130750ae-3ca5-48b4-b17a-151f361b31bc"
      },
      "outputs": [],
      "source": [
        "# traning model accuracy\n",
        "y_predict = []\n",
        "for x in X:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy:', sum(y_predict == np.array(y01))/len(np.array(y01)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gqx8ign8FBr8"
      },
      "source": [
        "The acciracy remained constant at $92\\%$"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5S5UyxQK5cKq"
      },
      "source": [
        "# Test Model\n",
        "\n",
        "Testing the model with unseen data."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fVScoBD75cKs",
        "outputId": "2f0f203e-74e0-4fa1-d36c-c54de5b65489"
      },
      "outputs": [],
      "source": [
        "target_o = [1 for i in range(25)]+[0 for i in range(25)]\n",
        "\n",
        "pathY=r'dataset/qso1/'\n",
        "pathN=r'dataset/noqso1/'\n",
        "nameN=''\n",
        "nameY=''\n",
        "inputY=[imageResize(callImage(i+1,pathY,nameY),16) for i in range(25)]\n",
        "inputN=[imageResize(callImage(i+1,pathN,nameN),16) for i in range(25)]\n",
        "input_combine = inputY+inputN\n",
        "\n",
        "np.random.seed(0)\n",
        "idx=np.array([int(i) for i in range(50)]).flatten()\n",
        "\n",
        "np.random.shuffle(idx)\n",
        "\n",
        "dataInput = list(input_combine[i] for i in idx )\n",
        "dataTarget = list( imageBinarize(input_combine[i]) for i in idx )\n",
        "\n",
        "data_target_o=list( target_o[i] for i in idx )\n",
        "\n",
        "Xtest= [normlaizeData(dataInput[i].flatten()) for i in range(25)]\n",
        "y01test= [data_target_o[i] for i in range(25)]\n",
        "\n",
        "Xtest1= [normlaizeData(dataInput[i].flatten()) for i in range(50)]\n",
        "y01test1= [data_target_o[i] for i in range(50)]\n",
        "\n",
        "y_predict = []\n",
        "for x in Xtest:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy25data:', sum(y_predict == np.array(y01test))/len(np.array(y01test)))\n",
        "\n",
        "y_predict1 = []\n",
        "for x in Xtest1:\n",
        "    output = model2(Tensor(x))\n",
        "    y_predict1 += [np.argmax(output.detach().numpy())]\n",
        "\n",
        "print('Accuracy50data:', sum(y_predict1 == np.array(y01test1))/len(np.array(y01test1)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "O6HXVKzwFU9t"
      },
      "source": [
        "The accuracy on different unseen data ranges from $80-84\\%$.\n",
        "\n",
        "\n",
        "We can say the model performs good on the unseen data too.\n",
        " \n",
        "\n",
        "We need new techniques to train the Hyperparameters to better fine tune them and reduce the computational time."
      ]
    }
  ],
  "metadata": {
    "colab": {
      "include_colab_link": true,
      "provenance": []
    },
    "interpreter": {
      "hash": "3bc52ef4732c2c89ac1c67f1a564bc39cd50f6fed15c43a2bb566bf993475a13"
    },
    "kernelspec": {
      "display_name": "Python 3.8.9 ('qc': venv)",
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
      "version": "3.11.4"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
