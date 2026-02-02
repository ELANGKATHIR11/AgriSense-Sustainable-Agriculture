const Joi = require('joi');

/**
 * Validation schemas for crop recommendation
 */
const cropRecommendationSchema = Joi.object({
    N: Joi.number().min(0).max(300).required()
        .messages({
            'number.base': 'Nitrogen (N) must be a number',
            'number.min': 'Nitrogen cannot be negative',
            'number.max': 'Nitrogen value too high (max 300)',
            'any.required': 'Nitrogen (N) is required'
        }),
    P: Joi.number().min(0).max(300).required()
        .messages({
            'number.base': 'Phosphorus (P) must be a number',
            'number.min': 'Phosphorus cannot be negative',
            'number.max': 'Phosphorus value too high (max 300)',
            'any.required': 'Phosphorus (P) is required'
        }),
    K: Joi.number().min(0).max(300).required()
        .messages({
            'number.base': 'Potassium (K) must be a number',
            'number.min': 'Potassium cannot be negative',
            'number.max': 'Potassium value too high (max 300)',
            'any.required': 'Potassium (K) is required'
        }),
    temperature: Joi.number().min(-10).max(60).required()
        .messages({
            'number.base': 'Temperature must be a number',
            'number.min': 'Temperature too low (min -10°C)',
            'number.max': 'Temperature too high (max 60°C)',
            'any.required': 'Temperature is required'
        }),
    humidity: Joi.number().min(0).max(100).required()
        .messages({
            'number.base': 'Humidity must be a number',
            'number.min': 'Humidity cannot be negative',
            'number.max': 'Humidity cannot exceed 100%',
            'any.required': 'Humidity is required'
        }),
    ph: Joi.number().min(0).max(14).required()
        .messages({
            'number.base': 'pH must be a number',
            'number.min': 'pH cannot be negative',
            'number.max': 'pH cannot exceed 14',
            'any.required': 'pH is required'
        }),
    rainfall: Joi.number().min(0).max(1000).required()
        .messages({
            'number.base': 'Rainfall must be a number',
            'number.min': 'Rainfall cannot be negative',
            'number.max': 'Rainfall value too high (max 1000mm)',
            'any.required': 'Rainfall is required'
        })
});

/**
 * Validation schema for yield prediction
 */
const yieldPredictionSchema = Joi.object({
    Area: Joi.number().min(0.1).max(10000).required()
        .messages({
            'number.base': 'Area must be a number',
            'number.min': 'Area must be at least 0.1 hectares',
            'number.max': 'Area too large (max 10000 hectares)',
            'any.required': 'Area is required'
        }),
    Item: Joi.string().min(2).max(50).required()
        .messages({
            'string.base': 'Crop name must be text',
            'string.min': 'Crop name too short',
            'string.max': 'Crop name too long',
            'any.required': 'Crop name is required'
        }),
    Year: Joi.number().integer().min(2000).max(2100).required()
        .messages({
            'number.base': 'Year must be a number',
            'number.min': 'Year too old',
            'number.max': 'Year too far in future',
            'any.required': 'Year is required'
        }),
    average_rain_fall_mm_per_year: Joi.number().min(0).max(5000).optional(),
    pesticides_tonnes: Joi.number().min(0).optional(),
    avg_temp: Joi.number().min(-20).max(50).optional()
});

/**
 * Validation schema for LLM queries
 */
const llmQuerySchema = Joi.object({
    query: Joi.string().min(3).max(1000).required()
        .messages({
            'string.base': 'Query must be text',
            'string.min': 'Query too short (min 3 characters)',
            'string.max': 'Query too long (max 1000 characters)',
            'any.required': 'Query is required'
        }),
    context: Joi.string().max(2000).optional()
});

/**
 * Validation schema for IoT device data
 */
const iotDeviceSchema = Joi.object({
    deviceId: Joi.string().alphanum().min(5).max(50).required(),
    deviceName: Joi.string().min(2).max(100).required(),
    location: Joi.string().min(2).max(200).optional(),
    sensors: Joi.array().items(
        Joi.object({
            type: Joi.string().valid('temperature', 'humidity', 'soil_moisture', 'ph', 'light').required(),
            value: Joi.number().required(),
            unit: Joi.string().required(),
            timestamp: Joi.date().iso().optional()
        })
    ).optional()
});

/**
 * Validation middleware factory
 */
const validate = (schema, property = 'body') => {
    return (req, res, next) => {
        const { error, value } = schema.validate(req[property], {
            abortEarly: false, // Get all errors, not just the first
            stripUnknown: true // Remove unknown fields
        });

        if (error) {
            const errors = error.details.map(detail => ({
                field: detail.path.join('.'),
                message: detail.message
            }));

            return res.status(400).json({
                error: 'Validation failed',
                message: 'Invalid input data',
                errors
            });
        }

        // Replace req[property] with validated value
        req[property] = value;
        next();
    };
};

module.exports = {
    validate,
    schemas: {
        cropRecommendation: cropRecommendationSchema,
        yieldPrediction: yieldPredictionSchema,
        llmQuery: llmQuerySchema,
        iotDevice: iotDeviceSchema
    }
};
