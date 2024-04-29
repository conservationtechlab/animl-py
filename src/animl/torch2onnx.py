"""
    Converts a given Torch model to an Onnx Model

    Author: Unduwap KandageDon
"""

import yaml
import torch 
import argparse

from animl.classifiers import EfficientNet

#used to create a model instance 
def load_model(architecture, num_classes):
    '''
        Creates a model instance. 
    '''
    if (architecture=="CTL"):
        model_instance = CTLClassifier(num_classes)
    elif (architecture=="efficientnet_v2_m"):
        model_instance = EfficientNet(num_classes,tune=False)        
    else:
        raise AssertionError('Please provide the correct model')

    return model_instance

#used to convert a model from a torch model path to onnx model 
def convert2onnx(architecture, num_classes, torchMPath, image_size=[299, 299], num_epochs=100, batch_size=32, outputOPath=None): 
    '''
        Converts a given torch model to an Onnx model

        Inputs: 
            architecture: Torch model architecture
            num_classes: number of label classes
            torchMPath: file path of the torch Model
            image_size: size of an image in training/test (Default = [299, 299])
            num_epochs: number of epochs used for model (Default = 100)
            batch_size: batch size for training (Default = 32)
            outputOPath: file path of output onnx file (Default = None)
        Outputs: None
        * the new model is saved to a given location, or by default to the same directory
    '''
    #initialize model based on given architecture
    modelInstance = load_model(architecture, num_classes)
    
    #load the torch model from the path
    pathModelAll = torch.load(torchMPath, map_location=torch.device('cpu'))
    #get the dict of the given model instance version of model from path 
    modelInstance.load_state_dict(pathModelAll["model"])
    #get the model ready
    modelInstance.eval()

  #changing from torch to onnx
    # RandomInput to the model
    RandomInput = torch.randn(batch_size, 3, image_size[0], image_size[1], requires_grad=True)
   
    #if an output path is not given, create a new path with the same name
    if outputOPath==None:
        outputOPath = torchMPath.replace(".pt", ".onnx")
        print(outputOPath)
        
    # Export the model to Onnx version
    torch.onnx.export(modelInstance,             # model being run
                      RandomInput,               # model input (or a tuple for multiple inputs)
                      outputOPath,               # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
def main(): 
    '''
    Command line function
    
    Example:
    > python torch2onnx.py --torchMPath /mnt/machinelearning/Models/Andes/Experiment2/30.pt 
                                                       --architecture efficientnet_v2_m --num_classes 53
    '''    
    #handle command line inputs
    parser = argparse.ArgumentParser(description='Convert a torch model into a onnx model.')
    parser.add_argument('--torchMPath', help='file path of the torch Model', required=True)
    parser.add_argument('--architecture', help='Torch model architecture', required=True)
    parser.add_argument('--num_classes', help='number of label classes', type=int, required=True)
    parser.add_argument('--image_size', help='size of an image in training/test', default=[299, 299], required=False)
    parser.add_argument('--num_epochs', help='number of epochs used for model', type=int, default=100, required=False)
    parser.add_argument('--batch_size', help='batch size for training', type=int, default=32, required=False)
    parser.add_argument('--outputOPath', help='file path of output onnx file', default=None, required=False)
    
    args = parser.parse_args()
    
    #Call convert2 onnx function with correct parameters
    convert2onnx(args.architecture, args.num_classes, args.torchMPath, args.image_size, 
                 args.num_epochs, args.batch_size, args.outputOPath)
        
#call main
if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
    