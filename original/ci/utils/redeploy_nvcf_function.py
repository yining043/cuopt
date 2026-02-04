# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import ngcsdk as ngc
import nvcf

# Acquire creds and details to be used for deployment
api_key = os.environ.get["CUOPT_PRD_NGC_API_KEY"]
org_name = os.environ.get["CUOPT_PRD_TEST_FUNCTION_ORG"]
func_id = os.environ.get("CUOPT_PRD_TEST_FUNCTION_ID", None)
ver_id = os.environ.get("CUOPT_PRD_TEST_FUNCTION_VERSION_ID", None)
backend = os.environ.get("CUOPT_NVCF_BACKEND", None)
instance_type = os.environ.get("CUOPT_NVCF_INSTANCE_TYPE", None)
gpu = os.environ.get("CUOPT_NVCF_GPU", "H100")

# Create a client
clt = ngc.Client(api_key=api_key)
clt.configure(api_key=api_key, org_name=org_name)

deployment_spec = nvcf.api.deployment_spec.DeploymentSpecification(
    backend=backend,
    gpu=gpu,
    min_instances=1,
    max_instances=1,
    instance_type=instance_type,
)

# Delete and Redeploy

deploy_clt = nvcf.api.deploy.DeployAPI(api_client=clt)

deploy_clt.delete(function_id=func_id, function_version_id=ver_id)

deploy_clt.create(
    function_id=func_id,
    function_version_id=ver_id,
    deployment_specifications=[deployment_spec],
)
