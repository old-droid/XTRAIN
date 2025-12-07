#!/usr/bin/env python

import subprocess
import sys

def run_command(command):
    try:
        print(f"Executing: {" ".join(command)}")
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("\n--- Command Output ---")
        print(process.stdout)
        if process.stderr:
            print("\n--- Command Errors ---")
            print(process.stderr)
        print("--- End Command Output ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError executing command: {" ".join(command)}")
        print(f"Return Code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Command not found. Make sure Python is in your PATH and scripts are executable.")

def main():
    print("Welcome to the XTRAIN CLI Tool!")
    print("---------------------------------")

    while True:
        print("\nPlease select an action:")
        print("1. Train Convolutional Neural Network (CNN)")
        print("2. Train Large Language Model (LLM)")
        print("3. Run an existing model (inference)")
        print("4. Exit")

        choice = None
        if len(sys.argv) > 1:
            choice = sys.argv[1].strip()
        else:
            choice = input("Enter your choice (1-4): ").strip()

        # If an argument was provided, and it's a valid choice, break after execution
        if len(sys.argv) > 1 and choice in ["1", "2", "3", "4"]:
            if choice == "4":
                print("Exiting XTRAIN CLI. Goodbye!")
                break

        if choice == "1":
            print("\nStarting CNN training...")
            run_command([sys.executable, "XTRAIN/train_cnn.py"])
            if len(sys.argv) > 1: break # Exit after executing if argument was provided
        elif choice == "2":
            print("\nStarting LLM training...")
            run_command([sys.executable, "XTRAIN/train_llm.py"])
            if len(sys.argv) > 1: break # Exit after executing if argument was provided
        elif choice == "3":
            print("\nStarting model inference...")
            run_command([sys.executable, "XTRAIN/run_model.py"])
            if len(sys.argv) > 1: break # Exit after executing if argument was provided
        elif choice == "4":
            print("Exiting XTRAIN CLI. Goodbye!")
            break
        else:
            # Only print invalid choice if running interactively or an invalid argument was given
            if len(sys.argv) <= 1 or choice not in ["1", "2", "3", "4"]:
                print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
