const WS = require('ws');
const axios = require('axios');

// Asterisk ARI endpoints and credentials
const AST_HOST = '127.0.0.1:8088';
const ARI_HTTP = `http://${AST_HOST}/ari`;
const WS_URL = `ws://${AST_HOST}/ari/events?app=voicebot&api_key=voicebot:supersecret`;

// Windows RTP gateway target
const WINDOWS_RTP_HOST = '172.18.176.1:4000'; // update if your Windows IP changes
const MEDIA_FORMAT = 'ulaw'; // 'ulaw' or 'alaw'

axios.defaults.auth = { username: 'voicebot', password: 'supersecret' };

const sessions = new Map();
const isExternal = (ch) => {
  const n = ch?.name || '';
  const t = ch?.channeltype || '';
  return n.startsWith('External/') || n.startsWith('UnicastRTP/') || t === 'UnicastRTP' || t === 'External';
};

async function safe(call, label) {
  try { return await call(); }
  catch (e) {
    const msg = e.response?.data || e.message;
    const code = e.response ? `HTTP ${e.response.status}` : '';
    console.error(`${label} failed:`, msg, code);
    throw e;
  }
}

async function cleanupByIds(extId, bridgeId) {
  if (extId) await safe(() => axios.delete(`${ARI_HTTP}/channels/${extId}`), 'DELETE ext').catch(() => {});
  if (bridgeId) await safe(() => axios.delete(`${ARI_HTTP}/bridges/${bridgeId}`), 'DELETE bridge').catch(() => {});
}

async function handleStart(ev) {
  const ch = ev.channel;
  if (isExternal(ch) || sessions.has(ch.id)) return;

  try {
    // Ensure media flows to the caller
    await safe(() => axios.post(`${ARI_HTTP}/channels/${ch.id}/answer`), 'POST answer');

    // Create external media first (direction both so we can send audio back)
    const emParams = {
      app: 'voicebot',
      external_host: WINDOWS_RTP_HOST,
      format: MEDIA_FORMAT,
      encapsulation: 'rtp',
      direction: 'both'
    };
    const ext = (await safe(() => axios.post(`${ARI_HTTP}/channels/externalMedia`, null, { params: emParams }), 'POST externalMedia')).data;

    // Create mixing bridge
    const bridge = (await safe(() => axios.post(`${ARI_HTTP}/bridges`, null, { params: { type: 'mixing' } }), 'POST bridges')).data;

    // Add both channels into bridge
    await safe(() => axios.post(`${ARI_HTTP}/bridges/${bridge.id}/addChannel`, null, { params: { channel: ch.id } }), 'add caller to bridge');
    await safe(() => axios.post(`${ARI_HTTP}/bridges/${bridge.id}/addChannel`, null, { params: { channel: ext.id } }), 'add ext to bridge');

    sessions.set(ch.id, { bridgeId: bridge.id, extId: ext.id });
    console.log(`Bridged ${ch.name} <-> External to ${WINDOWS_RTP_HOST} (${MEDIA_FORMAT})`);

  } catch (e) {
    // Best-effort cleanup
    const s = sessions.get(ch.id);
    await cleanupByIds(s?.extId, s?.bridgeId);
    sessions.delete(ch.id);
    try { await safe(() => axios.post(`${ARI_HTTP}/channels/${ch.id}/hangup`), 'hangup caller'); } catch {}
  }
}

async function handleEnd(ev) {
  const ch = ev.channel;
  const s = sessions.get(ch.id);
  if (!s) return;
  await cleanupByIds(s.extId, s.bridgeId);
  sessions.delete(ch.id);
}

const ws = new WS(WS_URL);
ws.on('open', () => console.log('ARI WS connected'));
ws.on('message', async (buf) => {
  let ev; try { ev = JSON.parse(buf); } catch { return; }
  if (ev.type === 'StasisStart') return handleStart(ev);
  if (ev.type === 'StasisEnd') return handleEnd(ev);
  if (ev.type === 'ChannelDestroyed') return handleEnd(ev);
});
ws.on('close', () => console.log('ARI WS closed'));
ws.on('error', (e) => console.error('ARI WS error', e.message));
