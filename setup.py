"""
Setup script for CPUWARP-ML framework
Compiles optimized C extensions for AMD and Intel CPUs
"""

from setuptools import setup, Extension
import numpy
import platform
import os

# Determine compiler flags based on platform and CPU
def get_compiler_flags():
    system = platform.system()
    machine = platform.machine().lower()
    
    flags = []
    
    # Base optimization flags
    if system == "Windows":
        # MSVC flags
        flags.extend([
            '/O2',        # Maximum optimization
            '/fp:fast',   # Fast floating point
            '/arch:AVX2', # AVX2 support
            '/openmp'     # OpenMP support
        ])
    else:
        # GCC/Clang flags
        flags.extend([
            '-O3',           # Maximum optimization
            '-ffast-math',   # Fast math optimizations
            '-march=native', # Use native CPU features
            '-mtune=native', # Tune for native CPU
            '-mavx2',        # AVX2 support
            '-mfma',         # FMA support
            '-fopenmp'       # OpenMP support
        ])
    
    # CPU-specific optimizations
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        vendor = cpu_info.get('vendor_id_raw', '').lower()
        
        if 'intel' in vendor:
            if system != "Windows":
                flags.append('-mtune=intel')
        elif 'amd' in vendor:
            if system != "Windows":
                flags.append('-mtune=znver2')  # Modern AMD Zen architecture
    except:
        pass
    
    return flags

# Define the C extension
def create_extension():
    include_dirs = [numpy.get_include()]
    
    # Add system-specific include directories
    if platform.system() == "Windows":
        # Add Windows SDK includes if available
        if 'INCLUDE' in os.environ:
            include_dirs.extend(os.environ['INCLUDE'].split(';'))
    
    libraries = []
    library_dirs = []
    
    # OpenMP library linking
    if platform.system() == "Windows":
        libraries.append('vcomp')  # MSVC OpenMP
    else:
        libraries.append('gomp')   # GCC OpenMP
    
    # Math library
    if platform.system() != "Windows":
        libraries.append('m')
    
    extension = Extension(
        name='optimized_kernels',
        sources=['optimized_kernels.c'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=get_compiler_flags(),
        language='c'
    )
    
    return extension

if __name__ == "__main__":
    # Install required packages first
    os.system("pip install numpy scipy psutil py-cpuinfo")
    
    setup(
        name="cpuwarp-ml",
        version="1.0.0",
        description="CPU-Optimized ML Training Framework with WARP",
        author="CPUWARP-ML Team",
        python_requires=">=3.7",
        install_requires=[
            'numpy>=1.19.0',
            'scipy>=1.5.0',
            'psutil>=5.7.0',
            'py-cpuinfo>=8.0.0'
        ],
        ext_modules=[create_extension()],
        py_modules=['cpuwarp_ml'],
        zip_safe=False,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: C",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
    
    print("\n" + "="*60)
    print("CPUWARP-ML Setup Complete!")
    print("="*60)
    print("CPU Architecture:", platform.machine())
    print("Compiler Flags:", " ".join(get_compiler_flags()))
    print("="*60)