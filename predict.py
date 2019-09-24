import torch
import numpy as np
import predict_args
import predict_utils

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
    img = predict_utils.process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()


    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return top_prob[0].numpy(), top_classes

def main():
    args = predict_args.get_args()
    print(args)
    
    image_path = args.image_path
    model, optimizer = predict_utils.load_checkpoint(args.checkpoint, args.arch, args.gpu)
    probs, classes = predict(image_path, model, args.top_k)
    cat_to_name = predict_utils.load_category_names(args.category_names)
    names = np.array([cat_to_name[i] for i in classes])
    for name, prob in zip(names,probs):
        print("{}: {}".format(name, str(prob)))
    
if __name__ == '__main__':
    main()
    