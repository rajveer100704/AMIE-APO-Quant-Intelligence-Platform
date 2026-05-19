import pytest
from unittest.mock import patch, MagicMock
from src.utils.distributed_cluster import init_distributed_cluster

@pytest.mark.unit
def test_init_distributed_cluster():
    with patch("src.utils.distributed_cluster.LocalCluster") as mock_cluster_class, \
         patch("src.utils.distributed_cluster.Client") as mock_client_class:
        
        mock_cluster = MagicMock()
        mock_cluster_class.return_value = mock_cluster
        mock_client = MagicMock()
        mock_client.dashboard_link = "http://localhost:8787/status"
        mock_client.scheduler_address = "tcp://localhost:8786"
        mock_client_class.return_value = mock_client
        
        client = init_distributed_cluster()
        
        assert client == mock_client
        mock_cluster_class.assert_called_once()
        mock_client_class.assert_called_once_with(mock_cluster)

@pytest.mark.unit
def test_init_distributed_cluster_exception():
    with patch("src.utils.distributed_cluster.LocalCluster", side_effect=Exception("Failed to start")):
        with pytest.raises(Exception) as exc_info:
            init_distributed_cluster()
        assert "Failed to start" in str(exc_info.value)
