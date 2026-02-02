const swaggerJsdoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');

const options = {
    definition: {
        openapi: '3.0.0',
        info: {
            title: 'AgriSense API Documentation',
            version: '1.0.0',
            description: 'Comprehensive API documentation for AgriSense - AI-Powered Agricultural Platform',
            contact: {
                name: 'AgriSense Team',
                email: 'support@agrisense.com'
            },
            license: {
                name: 'MIT',
                url: 'https://opensource.org/licenses/MIT'
            }
        },
        servers: [
            {
                url: 'http://localhost:5000',
                description: 'Development server'
            },
            {
                url: 'https://api.agrisense.com',
                description: 'Production server'
            }
        ],
        tags: [
            {
                name: 'ML Predictions',
                description: 'Machine Learning prediction endpoints'
            },
            {
                name: 'AI Services',
                description: 'LLM and VLM AI services'
            },
            {
                name: 'IoT',
                description: 'IoT device and sensor management'
            },
            {
                name: 'Health',
                description: 'Service health and monitoring'
            }
        ],
        components: {
            schemas: {
                CropRecommendationInput: {
                    type: 'object',
                    required: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
                    properties: {
                        N: {
                            type: 'number',
                            description: 'Nitrogen content (kg/ha)',
                            minimum: 0,
                            maximum: 300,
                            example: 90
                        },
                        P: {
                            type: 'number',
                            description: 'Phosphorus content (kg/ha)',
                            minimum: 0,
                            maximum: 300,
                            example: 42
                        },
                        K: {
                            type: 'number',
                            description: 'Potassium content (kg/ha)',
                            minimum: 0,
                            maximum: 300,
                            example: 43
                        },
                        temperature: {
                            type: 'number',
                            description: 'Temperature in Celsius',
                            minimum: -10,
                            maximum: 60,
                            example: 25
                        },
                        humidity: {
                            type: 'number',
                            description: 'Humidity percentage',
                            minimum: 0,
                            maximum: 100,
                            example: 80
                        },
                        ph: {
                            type: 'number',
                            description: 'Soil pH level',
                            minimum: 0,
                            maximum: 14,
                            example: 6.5
                        },
                        rainfall: {
                            type: 'number',
                            description: 'Rainfall in mm',
                            minimum: 0,
                            maximum: 1000,
                            example: 200
                        }
                    }
                },
                CropRecommendationResponse: {
                    type: 'object',
                    properties: {
                        success: {
                            type: 'boolean',
                            example: true
                        },
                        data: {
                            type: 'object',
                            properties: {
                                recommended_crop: {
                                    type: 'string',
                                    example: 'rice'
                                },
                                confidence: {
                                    type: 'number',
                                    example: 0.95
                                },
                                alternatives: {
                                    type: 'array',
                                    items: {
                                        type: 'object',
                                        properties: {
                                            crop: { type: 'string' },
                                            confidence: { type: 'number' }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                LLMQueryInput: {
                    type: 'object',
                    required: ['query'],
                    properties: {
                        query: {
                            type: 'string',
                            minLength: 3,
                            maxLength: 1000,
                            description: 'Agricultural question or query',
                            example: 'How to grow rice organically?'
                        },
                        context: {
                            type: 'object',
                            description: 'Additional context for the query'
                        }
                    }
                },
                Error: {
                    type: 'object',
                    properties: {
                        error: {
                            type: 'string',
                            example: 'Validation failed'
                        },
                        message: {
                            type: 'string',
                            example: 'Invalid input data'
                        },
                        errors: {
                            type: 'array',
                            items: {
                                type: 'object',
                                properties: {
                                    field: { type: 'string' },
                                    message: { type: 'string' }
                                }
                            }
                        }
                    }
                }
            },
            responses: {
                ValidationError: {
                    description: 'Validation error',
                    content: {
                        'application/json': {
                            schema: {
                                $ref: '#/components/schemas/Error'
                            }
                        }
                    }
                },
                UnauthorizedError: {
                    description: 'Unauthorized',
                    content: {
                        'application/json': {
                            schema: {
                                type: 'object',
                                properties: {
                                    error: { type: 'string', example: 'Unauthorized' }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    apis: ['./routes/*.js', './controllers/*.js'] // Path to API docs
};

const swaggerSpec = swaggerJsdoc(options);

module.exports = { swaggerUi, swaggerSpec };
