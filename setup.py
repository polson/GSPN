# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
from setuptools import setup, find_packages
from pathlib import Path
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

def make_cuda_ext(name, module, sources, include_dirs, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
        print(f'Compiling {sources} with CUDA')
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,)


if __name__ == '__main__':
    setup(
        name='gspn',
        version='0.1.0',
        description='Generalized Spatial Propagation Network - Parallel Sequence Modeling Framework',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        author='Hongjun Wang, Sifei Liu',
        author_email='',
        url='https://github.com/whj363636/GSPN',
        license='NVIDIA NC',
        packages=['gspn', 'ops', 'ops.gaterecurrent'],
        python_requires='>=3.8',
        install_requires=[
            'torch>=1.8.0',
        ],
        ext_modules=[
            make_cuda_ext(
                name='gaterecurrent2dnoind_cuda',
                module='ops.gaterecurrent',
                sources=['src/gaterecurrent2dnoind_cuda.cpp', 'src/gaterecurrent2dnoind_kernel.cu'],
                include_dirs=[Path(this_dir) / "ops" / "gaterecurrent"]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: Other/Proprietary License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
