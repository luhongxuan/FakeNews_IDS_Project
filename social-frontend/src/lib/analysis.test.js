import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeEvidence, recommendAction, riskLevel } from './analysis.js';

test('risk thresholds are deterministic', () => {
  assert.equal(riskLevel(80), 'high');
  assert.equal(riskLevel(50), 'medium');
  assert.equal(riskLevel(49.9), 'low');
});

test('low-confidence verdict remains unverified', () => {
  assert.equal(normalizeEvidence({ credibility: 'likely_false', confidence: 0.4 }).status, 'unverified');
  assert.equal(normalizeEvidence({ credibility: 'likely_true', confidence: 0.69 }).status, 'unverified');
});

test('confirmed true is never restricted', () => {
  assert.deepEqual(recommendAction({ risk: 'high', evidence: 'likely_true' }), {
    tier: 'none', label: '不限制，維持一般監測', strength: '0%',
  });
});

test('agent failure is not treated as misinformation', () => {
  assert.equal(recommendAction({ risk: 'high', evidence: 'agent_failure' }).tier, 'none');
});

test('high risk without evidence only receives a reversible soft suggestion', () => {
  assert.equal(recommendAction({ risk: 'high', evidence: 'unverified' }).tier, 'soft');
});

test('hard suggestion requires high risk and confirmed false evidence', () => {
  assert.equal(recommendAction({ risk: 'high', evidence: 'likely_false' }).tier, 'hard');
  assert.equal(recommendAction({ risk: 'low', evidence: 'likely_false' }).tier, 'none');
});
