"""
Azure Container Apps Autoscaling Configuration
Cost optimization with intelligent scaling rules
"""

# Azure autoscaling configuration for production deployment
autoscaling_config = {
    "api_version": "2023-05-01",
    "properties": {
        "configuration": {
            "activeRevisionsMode": "Single",
            "ingress": {
                "external": True,
                "targetPort": 8004,
                "transport": "auto",
                "allowInsecure": False,
                "traffic": [
                    {
                        "weight": 100,
                        "latestRevision": True
                    }
                ]
            },
            "dapr": {
                "enabled": False
            }
        },
        "template": {
            "containers": [
                {
                    "name": "agrisense-backend",
                    "image": "agrisenseacr.azurecr.io/agrisense-backend:latest",
                    "resources": {
                        "cpu": 0.5,
                        "memory": "1Gi"
                    },
                    "probes": [
                        {
                            "type": "liveness",
                            "httpGet": {
                                "path": "/health/live",
                                "port": 8004
                            },
                            "initialDelaySeconds": 10,
                            "periodSeconds": 30
                        },
                        {
                            "type": "readiness",
                            "httpGet": {
                                "path": "/health/ready",
                                "port": 8004
                            },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10
                        }
                    ],
                    "env": [
                        {
                            "name": "AGRISENSE_ENV",
                            "value": "production"
                        },
                        {
                            "name": "ENABLE_REDIS_CACHE",
                            "value": "true"
                        },
                        {
                            "name": "LOG_LEVEL",
                            "value": "INFO"
                        },
                        {
                            "name": "ENABLE_STRUCTURED_LOGGING",
                            "value": "true"
                        }
                    ]
                }
            ],
            "scale": {
                "minReplicas": 1,
                "maxReplicas": 10,
                "rules": [
                    {
                        "name": "http-rule",
                        "http": {
                            "metadata": {
                                "concurrentRequests": "50"
                            }
                        }
                    },
                    {
                        "name": "cpu-rule",
                        "custom": {
                            "type": "cpu",
                            "metadata": {
                                "type": "Utilization",
                                "value": "70"
                            }
                        }
                    },
                    {
                        "name": "memory-rule",
                        "custom": {
                            "type": "memory",
                            "metadata": {
                                "type": "Utilization",
                                "value": "80"
                            }
                        }
                    }
                ]
            }
        }
    }
}

# Cost optimization settings
cost_optimization = {
    "description": "Cost optimization configuration for AgriSense",
    
    # Development environment (minimal cost)
    "development": {
        "minReplicas": 1,
        "maxReplicas": 2,
        "cpu": 0.25,
        "memory": "0.5Gi",
        "storage_tier": "hot_only",
        "estimated_monthly_cost_usd": 15
    },
    
    # Staging environment (moderate cost)
    "staging": {
        "minReplicas": 1,
        "maxReplicas": 5,
        "cpu": 0.5,
        "memory": "1Gi",
        "storage_tier": "hot_30_days",
        "estimated_monthly_cost_usd": 50
    },
    
    # Production environment (optimized cost)
    "production": {
        "minReplicas": 2,  # For high availability
        "maxReplicas": 10,
        "cpu": 0.5,
        "memory": "1Gi",
        "storage_tier": "hot_30_cold_archive",
        "estimated_monthly_cost_usd": 120
    },
    
    # Storage tiering strategy
    "storage_tiering": {
        "hot_storage_days": 30,
        "cool_storage_days": 90,
        "archive_storage_days": 365,
        
        "retention_policy": {
            "sensor_readings": {
                "hot": "30 days - frequent access",
                "cool": "31-90 days - occasional access",
                "archive": "91+ days - compliance/historical"
            },
            "predictions": {
                "hot": "7 days",
                "cool": "8-30 days",
                "archive": "31+ days"
            },
            "logs": {
                "hot": "14 days",
                "cool": "15-60 days",
                "archive": "61+ days"
            }
        }
    },
    
    # Cost monitoring alerts
    "cost_alerts": {
        "daily_budget_usd": 5,
        "monthly_budget_usd": 150,
        "alert_threshold_percent": 80,
        "alert_emails": [
            "admin@agrisense.ai"
        ]
    }
}

# Autoscaling schedule (time-based scaling)
scaling_schedule = {
    "description": "Time-based autoscaling for predictable load patterns",
    
    "schedules": [
        {
            "name": "business_hours",
            "timezone": "Asia/Kolkata",
            "start": "06:00",
            "end": "20:00",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            "minReplicas": 2,
            "maxReplicas": 10
        },
        {
            "name": "off_hours",
            "timezone": "Asia/Kolkata",
            "start": "20:00",
            "end": "06:00",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "minReplicas": 1,
            "maxReplicas": 3
        },
        {
            "name": "weekend",
            "timezone": "Asia/Kolkata",
            "start": "00:00",
            "end": "23:59",
            "days": ["Sunday"],
            "minReplicas": 1,
            "maxReplicas": 2
        }
    ]
}

# Celery worker autoscaling
celery_autoscaling = {
    "description": "Celery worker autoscaling for ML tasks",
    
    "worker_config": {
        "minWorkers": 1,
        "maxWorkers": 5,
        "taskConcurrency": 2,  # Tasks per worker
        
        "scaling_rules": [
            {
                "name": "queue-length",
                "type": "queue",
                "metadata": {
                    "queueName": "ml_predictions",
                    "queueLength": "20",
                    "scaleUpThreshold": "15",
                    "scaleDownThreshold": "5"
                }
            },
            {
                "name": "cpu-usage",
                "type": "cpu",
                "metadata": {
                    "type": "Utilization",
                    "value": "70"
                }
            }
        ]
    },
    
    "queues": {
        "high_priority": {
            "name": "ml_predictions_high",
            "routingKey": "ml.predict.high",
            "workers": 2,
            "concurrency": 1
        },
        "normal_priority": {
            "name": "ml_predictions",
            "routingKey": "ml.predict.normal",
            "workers": 3,
            "concurrency": 2
        },
        "low_priority": {
            "name": "ml_train",
            "routingKey": "ml.train",
            "workers": 1,
            "concurrency": 1
        }
    }
}

# Monitoring and alerting
monitoring_config = {
    "description": "Azure Monitor configuration",
    
    "metrics": [
        {
            "name": "request_count",
            "type": "counter",
            "alert_threshold": 1000,
            "time_window": "5m"
        },
        {
            "name": "response_time_p95",
            "type": "histogram",
            "alert_threshold": 2000,  # ms
            "time_window": "5m"
        },
        {
            "name": "error_rate",
            "type": "gauge",
            "alert_threshold": 5,  # percent
            "time_window": "5m"
        },
        {
            "name": "replica_count",
            "type": "gauge",
            "alert_threshold": 8,  # approaching max
            "time_window": "5m"
        }
    ],
    
    "log_analytics": {
        "enabled": True,
        "retention_days": 30,
        "daily_cap_gb": 1
    },
    
    "application_insights": {
        "enabled": True,
        "sampling_percentage": 10,  # Sample 10% of requests
        "track_dependencies": True,
        "track_exceptions": True
    }
}
