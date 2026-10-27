import torch # Import torch

def main():
    print("--- Esercizio 2 & 3 ---")
    # 2. Create a random tensor with shape (7, 7)
    random_tensor = torch.rand(7, 7)
    
    # Create another random tensor with shape (1, 7)
    random_tensor2 = torch.rand(1, 7)
    
    # 3. Perform matrix multiplication (with transpose)
    multiplication = random_tensor @ random_tensor2.T
    print("Prodotto:", multiplication)


    print("\n--- Esercizio 4 (Seed) ---")
    # 4. Set manual seed to 0
    torch.manual_seed(0)
    
    # Create two random tensors again
    random_tensor = torch.rand(7, 7)
    random_tensor2 = torch.rand(1, 7)
    
    # Matrix multiply tensors
    multiplication = random_tensor @ random_tensor2.T
    print("Prodotto con seed:", multiplication)


    print("\n--- Esercizio 5, 6, 7, 8 & 9 (GPU) ---")
    # Set random seed
    torch.manual_seed(1234)
    
    # 6. Create two random tensors of shape (2, 3)
    random_tensor3 = torch.rand(2, 3)
    random_tensor4 = torch.rand(2, 3)

    # Check for access to GPU
    if torch.cuda.is_available():
        # 5. Set random seed on the GPU
        torch.cuda.manual_seed(1234)
        
        # Send them both to the GPU
        random_tensor3 = random_tensor3.to('cuda')
        random_tensor4 = random_tensor4.to('cuda')
        
        # 7. Perform matmul on tensor_A and tensor_B
        multiplication2 = random_tensor3 @ random_tensor4.T
        print("Prodotto su GPU:\n", multiplication2)
        
        # 8. Find max and min
        print("Max:", multiplication2.max())
        print("Min:", multiplication2.min())
        
        # 9. Find arg max and arg min
        print("Argmax:", torch.argmax(multiplication2))
        print("Argmin:", torch.argmin(multiplication2))
    else:
        print("GPU non disponibile.")


    print("\n--- Esercizio 10 (Squeeze) ---")
    # 10. Set seed to 7
    torch.manual_seed(7)
    
    # Make a random tensor with shape (1, 1, 1, 10)
    random_tensor5 = torch.rand(1, 1, 1, 10)
    
    # Remove single dimensions
    random_tensor6 = random_tensor5.squeeze()
    
    # Print out tensors and their shapes
    print("Original shape:", random_tensor5.size())
    print("Squeezed shape:", random_tensor6.size())

if __name__ == "__main__":
    main()