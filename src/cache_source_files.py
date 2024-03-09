# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import os
import shutil
from pathlib import Path
from typing import Set


def get_referenced_files(current_file: str) -> Set[str]:
    project_directory = os.path.dirname(current_file)

    with open(current_file, 'r') as file:
        tree = ast.parse(file.read())

    referenced_files = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                module_name = os.path.join(*name.name.split('.')) + '.py'
                abs_filename = os.path.join(project_directory, module_name)
                if os.path.exists(abs_filename):
                    referenced_files.add(module_name)
                    referenced_files = referenced_files.union(get_referenced_files(abs_filename))
        elif isinstance(node, ast.ImportFrom):
            module_name = os.path.join(*node.module.split('.')) + '.py'
            abs_filename = os.path.join(project_directory, module_name)
            if os.path.exists(abs_filename):
                referenced_files.add(module_name)
                referenced_files = referenced_files.union(get_referenced_files(abs_filename))
    return referenced_files


def copy_referenced_files_to(root_file: str, cache_dir: str):
    project_directory = os.path.dirname(os.path.abspath(root_file))
    for file in get_referenced_files(root_file):
        dump_path = os.path.join(cache_dir, file)
        Path(os.path.dirname(dump_path)).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(project_directory, file), dump_path)
