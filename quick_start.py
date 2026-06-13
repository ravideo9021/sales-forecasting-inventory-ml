#!/usr/bin/env python3
"""
Quick Start Script for Sales Forecasting & Inventory Optimization ML Project.

This script provides an easy way to get started with the project by:
1. Setting up the environment
2. Downloading sample data (or creating synthetic data)
3. Running the pipeline
4. Launching the dashboard

Usage:
    python quick_start.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║        Sales Forecasting & Inventory Optimization ML Project     ║
    ║                           Quick Start                            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version check passed: {sys.version.split()[0]}")


def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True


def install_dependencies():
    """Install project dependencies."""
    print("📚 Installing dependencies...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip.exe"
    else:  # Unix/MacOS
        pip_path = "venv/bin/pip"
    
    try:
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True, capture_output=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_directories():
    """Set up project directories."""
    print("📁 Setting up project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "data/synthetic",
        "models",
        "reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")


def run_pipeline():
    """Run the ML pipeline."""
    print("🚀 Running the ML pipeline...")
    
    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python.exe"
    else:  # Unix/MacOS
        python_path = "venv/bin/python"
    
    try:
        # Run the main pipeline
        result = subprocess.run([python_path, "main.py", "--phase", "all"], 
                              capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            print("\n📊 Pipeline Results:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ Pipeline failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Pipeline timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False


def launch_dashboard(background=True):
    """Launch the Streamlit dashboard."""
    print("🎛️ Launching the dashboard...")
    
    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python.exe"
        streamlit_path = "venv\\Scripts\\streamlit.exe"
    else:  # Unix/MacOS
        python_path = "venv/bin/python"
        streamlit_path = "venv/bin/streamlit"
    
    try:
        if background:
            # Launch in background
            process = subprocess.Popen([streamlit_path, "run", "app.py", "--server.port", "8501"])
            print("✅ Dashboard launched successfully!")
            print("🌐 Dashboard URL: http://localhost:8501")
            print("📝 Note: Dashboard is running in the background")
            return process
        else:
            # Launch in foreground (blocking)
            subprocess.run([streamlit_path, "run", "app.py", "--server.port", "8501"])
            
    except FileNotFoundError:
        print("❌ Streamlit not found. Trying alternative method...")
        try:
            subprocess.run([python_path, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
        except Exception as e:
            print(f"❌ Failed to launch dashboard: {e}")
            return None
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return None


def print_next_steps():
    """Print next steps for the user."""
    next_steps = """
    🎉 Setup Complete! Next Steps:
    
    1. 📊 Dashboard: Your interactive dashboard is running at http://localhost:8501
    2. 📁 Data: Check the 'data/processed/' folder for generated datasets
    3. 🤖 Models: Trained models are saved in the 'models/' folder
    4. 📈 Reports: Analysis reports are in the 'reports/' folder
    5. 📋 Logs: Check 'logs/' folder for detailed execution logs
    
    🔧 Advanced Usage:
    - Run specific phases: python main.py --phase train
    - Customize configuration: Edit config/config.yaml
    - Add your own data: Place CSV files in data/raw/
    
    📚 Documentation: Check README.md for detailed instructions
    
    Happy forecasting! 🚀
    """
    print(next_steps)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick Start for Sales Forecasting ML Project')
    parser.add_argument('--skip-pipeline', action='store_true', 
                       help='Skip running the ML pipeline and only launch dashboard')
    parser.add_argument('--skip-dashboard', action='store_true',
                       help='Skip launching the dashboard')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Step 3: Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Step 4: Setup directories
    setup_directories()
    
    # Step 5: Run pipeline (unless skipped)
    if not args.skip_pipeline:
        if not run_pipeline():
            print("⚠️ Pipeline failed, but continuing with dashboard setup...")
    else:
        print("⏭️ Skipping pipeline execution")
    
    # Step 6: Launch dashboard (unless skipped)
    if not args.skip_dashboard:
        dashboard_process = launch_dashboard(background=True)
        
        if dashboard_process:
            # Wait a moment for the dashboard to start
            time.sleep(3)
            print_next_steps()
            
            # Keep the script running to maintain the dashboard
            try:
                print("\n💡 Press Ctrl+C to stop the dashboard and exit")
                dashboard_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping dashboard...")
                dashboard_process.terminate()
                print("✅ Dashboard stopped")
        else:
            print("❌ Dashboard failed to launch")
    else:
        print("⏭️ Skipping dashboard launch")
        print_next_steps()


if __name__ == "__main__":
    main() 