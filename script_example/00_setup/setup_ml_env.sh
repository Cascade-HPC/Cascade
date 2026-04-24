#!/bin/bash
source "$(dirname "$0")/setup_env.sh"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
