const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

let jobs = {}; // in-memory job store: { jobId: { status, progress, modelName, datasetId, details } }

const createJob = async ({ modelName, datasetId = null, meta = {} } = {}) => {
  const id = uuidv4();
  const job = {
    id,
    modelName,
    datasetId,
    status: 'queued',
    progress: 0,
    details: meta,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };
  jobs[id] = job;

  // Attempt to persist to Postgres if available
  try {
    const pg = require('../config/postgres');
    if (pg && pg.pool) {
      await pg.pool.query(
        `INSERT INTO ml_jobs(id, model_name, dataset_id, status, progress, details, created_at, updated_at) VALUES($1,$2,$3,$4,$5,$6,$7,$8)`,
        [id, modelName, datasetId, job.status, job.progress, job.details || null, job.created_at, job.updated_at]
      );
    }
  } catch (e) {
    logger.warn('Could not persist ml_job to Postgres', { error: e.message });
  }

  return job;
};

const updateJob = async (id, patch = {}) => {
  if (!jobs[id]) return null;
  jobs[id] = { ...jobs[id], ...patch, updated_at: new Date().toISOString() };
  // Persist
  try {
    const pg = require('../config/postgres');
    if (pg && pg.pool) {
      const fields = [];
      const vals = [];
      let i = 1;
      for (const k of Object.keys(patch)) {
        fields.push(`${k} = $${i}`);
        vals.push(patch[k]);
        i++;
      }
      if (fields.length > 0) {
        vals.push(id);
        await pg.pool.query(`UPDATE ml_jobs SET ${fields.join(',')}, updated_at = now() WHERE id = $${i}`, vals);
      }
    }
  } catch (e) {
    logger.warn('Could not update ml_job in Postgres', { error: e.message });
  }
  return jobs[id];
};

const getJob = (id) => jobs[id] || null;

module.exports = { createJob, updateJob, getJob };
