# Kubernetes Scheduling

The Kubernetes scheduler assigns newly created Pods to suitable worker nodes. A Pod remains pending until the scheduler selects a node and records the binding through the Kubernetes API.

Scheduling begins by filtering out nodes that cannot run the Pod. A node may be excluded because it lacks sufficient CPU or memory, does not satisfy a node selector or required node affinity rule, contains an incompatible taint, or cannot meet a storage requirement.

The scheduler then scores the remaining feasible nodes. Scoring considers preferences such as balanced resource usage, preferred affinity, topology distribution, and the placement of related workloads. The highest-ranked node is normally selected.

Requests describe the CPU and memory that the scheduler reserves for a container. Limits constrain the resources a running container may consume. Accurate requests are important because the scheduler bases placement decisions on requested resources rather than current instantaneous usage.

Taints allow nodes to repel Pods, while tolerations permit specific Pods to be scheduled onto those nodes. Node affinity attracts Pods to nodes with matching labels. Pod affinity and anti-affinity control whether workloads should run near or away from other Pods.

The scheduler chooses a node but does not start containers itself. After a Pod is bound, the kubelet on the selected node works with the container runtime to obtain images and start the containers.
