"""
网络接口配置 | Network Interface Configuration

自动配置以太网接口 IP 和路由，替代手动运行 shell 脚本。
需要 root 权限（或 sudo 免密）来执行 ip 命令。
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def _run(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    """执行 shell 命令并返回结果。"""
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=check,
    )


def ensure_interface(
    iface: str,
    local_ip: str,
    target_ip: str,
    delete_default_route: bool = False,
) -> bool:
    """
    确保网络接口已配置并可达目标 IP。

    Args:
        iface: 网络接口名 (e.g. "eth0")
        local_ip: 本机 IP/掩码 (e.g. "192.168.1.132/24")
        target_ip: 目标设备 IP (e.g. "192.168.1.133")
        delete_default_route: 是否删除该接口的默认路由（防止与 VPN 冲突）

    Returns:
        True 如果目标可达，False 否则（仅警告，不抛异常）
    """
    # 1. 启用接口
    r = _run(f"sudo ip link set {iface} up")
    if r.returncode != 0:
        logger.warning("无法启用接口 %s: %s", iface, r.stderr.strip())
        return False
    logger.info("接口 %s 已启用", iface)

    # 2. 检查 IP 是否已分配，没有则添加
    ip_only = local_ip.split('/')[0]
    r = _run(f"ip addr show {iface}")
    if ip_only not in r.stdout:
        r = _run(f"sudo ip addr add {local_ip} dev {iface}")
        if r.returncode != 0:
            # RTNETLINK "File exists" 表示 IP 已存在（可能带不同掩码），可忽略
            if "File exists" not in r.stderr:
                logger.warning("添加 IP %s 到 %s 失败: %s", local_ip, iface, r.stderr.strip())
                return False
        logger.info("已分配 %s 到 %s", local_ip, iface)
    else:
        logger.debug("IP %s 已在 %s 上", ip_only, iface)

    # 3. 添加 host route：确保目标 IP 走指定接口
    r = _run(f"sudo ip route replace {target_ip}/32 dev {iface}")
    if r.returncode != 0:
        logger.warning("添加 host route 失败: %s", r.stderr.strip())
    else:
        logger.info("Host route: %s/32 -> %s", target_ip, iface)

    # 4. 可选：删除该接口的默认路由（保护 VPN / 其他默认网关）
    if delete_default_route:
        r = _run(f"sudo ip route del default dev {iface} 2>/dev/null")
        if r.returncode == 0:
            logger.info("已删除 %s 默认路由", iface)

    # 5. ping 验证可达性
    r = _run(f"ping -c 1 -W 2 -I {iface} {target_ip}")
    if r.returncode == 0:
        logger.info("网络配置完成: %s -> %s 可达", iface, target_ip)
        return True
    else:
        logger.warning("ping %s 失败（接口 %s），设备可能未开启", target_ip, iface)
        return False
