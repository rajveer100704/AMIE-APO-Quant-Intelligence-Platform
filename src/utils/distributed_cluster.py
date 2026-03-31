import os
import psutil
from distributed import Client, LocalCluster
import structlog

logger = structlog.get_logger(__name__)

def init_distributed_cluster():
    """
    Initializes a local Dask Distributed cluster with dynamic resource allocation.
    Leaves 1 core free for system stability.
    """
    try:
        # Dynamic CPU allocation
        total_cpus = psutil.cpu_count(logical=True)
        # Use total_cpus - 1, minimum 1
        n_workers = max(1, total_cpus - 1)
        
        logger.info("Initializing Dask cluster", 
                    total_cpus=total_cpus, 
                    allocated_workers=n_workers)
        
        # Initialize LocalCluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit='auto',
            dashboard_address=':8787'
        )
        
        client = Client(cluster)
        logger.info("Dask cluster ready", 
                    dashboard_url=client.dashboard_link,
                    scheduler_address=client.scheduler_address)
        
        return client
    except Exception as e:
        logger.error("Failed to initialize Dask cluster", error=str(e))
        # Fallback to synchronous execution or raise
        raise

if __name__ == "__main__":
    # Basic test
    client = init_distributed_cluster()
    print(f"Cluster Dashboard: {client.dashboard_link}")
    client.close()
