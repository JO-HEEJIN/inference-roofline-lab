# Ascend access decision — 2026-09-17

User preference: lowest total cost. No account created, payment made, or NPU allocated.

First choice: CANNLab's free allowance, subject to account eligibility and capacity.
The [CANN community manual](https://gitcode.com/cann/infrastructure/blob/main/docs/CANNLab/CANNLab%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C.md)
lists 100 initial NPU card-hours, A2 single-card instances, preinstalled CANN,
WebIDE and remote IDE access. First use requires GitCode/Huawei Cloud account
setup and real-name verification. International-user eligibility is unconfirmed.
Select physical A2 NPU resources, not the CPU simulator. Verify the actual chip,
toolchain compatibility and access permissions before building the pinned source.
Use `/mnt/workspace` for persistence across shutdown; back up before deletion.
The public repository entry currently presents a login panel; user login is needed.

Rejected for this task: [CloudGPU documentation](https://cloudgpu.app/docs)
says raw SSH rentals are not yet available. Its
[$0.95/hour 910B advertisement](https://cloudgpu.app/pricing) therefore does not
establish access suitable for compiling this custom kernel.

Unverified alternative: [Yissou product listing](https://www.yissou.com/products)
advertises CNY 3.50/hour, but [its homepage](https://yissou.com/)
shows conflicting dollar pricing and no stock. Do not treat this as a confirmed
offer or send funds based on the advertisement.

Decision: try the free official development environment first. If identity or
availability prevents access, obtain a concrete single-card rental quote with
terminal/compiler access, deposit and storage costs before requesting payment.
