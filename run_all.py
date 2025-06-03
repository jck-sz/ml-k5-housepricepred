import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        # Run the command
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True,
            capture_output=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Error message: {e.stderr}")
        return False

def main():
    """Run the complete ML pipeline."""
    print("🏠 HOUSE PRICE PREDICTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src") or not os.path.exists("app"):
        print("❌ Error: Please run this script from the project root directory.")
        print("   Current directory:", os.getcwd())
        return
    
    # Step 1: Preprocess data
    if not run_command(
        f"{sys.executable} src/data_preprocessing/preprocess.py",
        "Step 1: Data Preprocessing"
    ):
        print("\n❌ Pipeline stopped due to preprocessing error.")
        return
    
    time.sleep(2)  # Brief pause between steps
    
    # Step 2: Train model
    if not run_command(
        f"{sys.executable} src/models/train_model.py",
        "Step 2: Model Training"
    ):
        print("\n❌ Pipeline stopped due to training error.")
        return
    
    time.sleep(2)
    
    # Step 3: Launch Streamlit app
    print(f"\n{'='*60}")
    print("🚀 Step 3: Launching Streamlit App")
    print(f"{'='*60}")
    print("\n📌 The app will open in your default web browser.")
    print("📌 To stop the app, press Ctrl+C in this terminal.\n")
    
    try:
        # Run Streamlit app (this will keep running until user stops it)
        subprocess.run(
            f"streamlit run app/app.py",
            shell=True,
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Streamlit app stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching Streamlit app: {e}")

if __name__ == "__main__":
    main()